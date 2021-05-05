package is.hail.expr.ir

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend.BackendContext
import is.hail.expr.ir.Emit.E
import is.hail.expr.ir.agg.{AggStateSig, ArrayAggStateSig, GroupedStateSig}
import is.hail.expr.ir.functions.StringFunctions
import is.hail.expr.ir.lowering.TableStageDependency
import is.hail.expr.ir.streams.{EmitStream, StreamProducer, StreamUtils}
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer}
import is.hail.linalg.{BLAS, LAPACK, LinalgCodeUtils}
import is.hail.services.shuffler._
import is.hail.types.physical._
import is.hail.types.physical.stypes.concrete.{SBaseStructPointerCode, SCanonicalShufflePointerCode, SCanonicalShufflePointerSettable}
import is.hail.types.physical.stypes.interfaces.{SBaseStructCode, SNDArray, SNDArrayCode, SStreamCode}
import is.hail.types.physical.stypes.primitives.{SFloat32, SFloat64, SInt32, SInt32Code, SInt64}
import is.hail.types.physical.stypes.{EmitType, SCode, SType}
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.utils.richUtils.RichCodeRegion

import java.io._
import scala.collection.mutable
import scala.language.{existentials, postfixOps}

// class for holding all information computed ahead-of-time that we need in the emitter
class EmitContext(val executeContext: ExecuteContext, val req: RequirednessAnalysis)


object SetupBuilder {
  def apply(mb: EmitMethodBuilder[_]): SetupBuilder = new SetupBuilder(mb, Code._empty)

  def apply(mb: EmitMethodBuilder[_], setup: Code[Unit]): SetupBuilder = new SetupBuilder(mb, setup)

  def map[T, U](mb: EmitMethodBuilder[_])(is: IndexedSeq[T])(f: (SetupBuilder, T) => U): (Code[Unit], IndexedSeq[U]) = {
    val sb = SetupBuilder(mb)
    val rs = sb.map(is)(f)
    (sb.setup, rs)
  }

  def map[T, U](mb: EmitMethodBuilder[_], setup: Code[Unit])(is: IndexedSeq[T])(f: (SetupBuilder, T) => U): (Code[Unit], IndexedSeq[U]) = {
    val sb = SetupBuilder(mb, setup)
    val rs = sb.map(is)(f)
    (sb.setup, rs)
  }
}

class SetupBuilder(mb: EmitMethodBuilder[_], var setup: Code[Unit]) {
  def append(c: Code[Unit]): Unit = {
    setup = Code(setup, c)
  }

  def +=(c: Code[Unit]): Unit = append(c)

  def memoize[T](e: Code[T], name: String)(implicit tti: TypeInfo[T]): Value[T] = {
    val l = mb.newLocal[T](name)
    append(l := e)
    l
  }

  def memoizeField[T](e: Code[T], name: String)(implicit tti: TypeInfo[T]): Value[T] = {
    val l = mb.genFieldThisRef[T](name)
    append(l := e)
    l
  }

  def map[T, U](is: IndexedSeq[T])(f: (SetupBuilder, T) => U): IndexedSeq[U] = is.map(f(this, _))

  def result(): Code[Unit] = {
    val r = setup
    setup = null
    r
  }
}

object Emit {
  type E = Env[EmitValue]

  def apply[C](ctx: EmitContext, ir: IR, fb: EmitFunctionBuilder[C], rti: TypeInfo[_], aggs: Option[Array[AggStateSig]] = None): Option[SingleCodeType] = {
    TypeCheck(ir)

    val mb = fb.apply_method
    val container = aggs.map { a =>
      val c = fb.addAggStates(a)
      AggContainer(a, c, () => ())
    }
    val emitter = new Emit[C](ctx, fb.ecb)
    val region = mb.getCodeParam[Region](1)
    val returnTypeOption: Option[SingleCodeType] = if (ir.typ == TVoid) {
      fb.apply_method.voidWithBuilder { cb =>
        emitter.emitVoid(cb, ir, region, Env.empty, container, None)
      }
      None
    } else {
      var sct: SingleCodeType = null
      fb.emitWithBuilder { cb =>

        val pcode = emitter.emitI(ir, cb, region, Env.empty, container, None).handle(cb, {
          cb._throw[RuntimeException](
            Code.newInstance[RuntimeException, String]("cannot return empty"))
        })

        val scp = SingleCodePCode.fromPCode(cb, pcode, region)
        assert(scp.typ.ti == rti, s"type info mismatch: expect $rti, got ${ scp.typ.ti }")
        sct = scp.typ
        scp.code
      }
      Some(sct)
    }
    returnTypeOption
  }
}

object AggContainer {
  // FIXME remove this when EmitStream also has a codebuilder
  def fromVars(aggs: Array[AggStateSig], mb: EmitMethodBuilder[_], region: Settable[Region], off: Settable[Long]): (AggContainer, EmitCodeBuilder => Unit, EmitCodeBuilder => Unit) = {

    val (setup, aggState) = EmitCodeBuilder.scoped(mb) { cb =>
      val states = agg.StateTuple(aggs.map(a => agg.AggStateSig.getState(a, cb.emb.ecb)))
      val aggState = new agg.TupleAggregatorState(mb.ecb, states, region, off)
      cb += (region := Region.stagedCreate(Region.REGULAR, cb.emb.ecb.pool()))
      cb += region.load().setNumParents(aggs.length)
      cb += (off := region.load().allocate(aggState.storageType.alignment, aggState.storageType.byteSize))
      states.createStates(cb)
      aggState
    }

    val cleanup =  { cb: EmitCodeBuilder =>
      aggState.store(cb)
      cb += region.load().invalidate()
      cb.assign(region, Code._null)
    }

    (AggContainer(aggs, aggState, () => ()), (cb: EmitCodeBuilder) => cb += setup, cleanup)
  }

  def fromMethodBuilder[C](aggs: Array[AggStateSig], mb: EmitMethodBuilder[C], varPrefix: String): (AggContainer, EmitCodeBuilder => Unit, EmitCodeBuilder => Unit) =
    fromVars(aggs, mb, mb.genFieldThisRef[Region](s"${varPrefix}_top_region"), mb.genFieldThisRef[Long](s"${varPrefix}_off"))

  def fromBuilder[C](cb: EmitCodeBuilder, aggs: Array[AggStateSig], varPrefix: String): AggContainer = {
    val off = cb.newField[Long](s"${varPrefix}_off")
    val region = cb.newField[Region](s"${varPrefix}_top_region", Region.stagedCreate(Region.REGULAR, cb.emb.ecb.pool()))
    val states = agg.StateTuple(aggs.map(a => agg.AggStateSig.getState(a, cb.emb.ecb)))
    val aggState = new agg.TupleAggregatorState(cb.emb.ecb, states, region, off)
    cb += region.load().setNumParents(aggs.length)
    cb.assign(off, region.load().allocate(aggState.storageType.alignment, aggState.storageType.byteSize))
    states.createStates(cb)

    AggContainer(aggs, aggState, { () =>
      aggState.store(cb)
      cb += region.load().invalidate()
      cb.assign(region, Code._null)
    })
  }
}

case class AggContainer(aggs: Array[AggStateSig], container: agg.TupleAggregatorState, cleanup: () => Unit) {

  def nested(i: Int, init: Boolean): Option[AggContainer] = {
    aggs(i).n.map { nested =>
      val c = aggs(i) match {
        case _: GroupedStateSig =>
          val state = container.states(i).asInstanceOf[agg.DictState]
          if (init) state.initContainer else state.keyed.container
        case _: ArrayAggStateSig =>
          val state = container.states(i).asInstanceOf[agg.ArrayElementState]
          if (init) state.initContainer else state.container
      }
      AggContainer(nested.toArray, c, () => ())
    }
  }
}

object EmitRegion {
  def default(mb: EmitMethodBuilder[_]): EmitRegion = EmitRegion(mb, mb.getCodeParam[Region](1))
}

case class EmitRegion(mb: EmitMethodBuilder[_], region: Value[Region]) {
  def baseRegion: Value[Region] = mb.getCodeParam[Region](1)
}

abstract class EmitValue {
  def pt: PType

  def load: EmitCode

  def get(cb: EmitCodeBuilder): PCode
}

class EmitUnrealizableValue(private val ec: EmitCode) extends EmitValue {
  val pt: PType = ec.pt
  assert(!pt.isRealizable)
  private[this] var used: Boolean = false

  def load: EmitCode = {
    assert(!used)
    used = true
    ec
  }

  override def get(cb: EmitCodeBuilder): PCode = throw new UnsupportedOperationException(s"Can't make PValue for unrealizable type ${pt}")
}

/**
 * Notes on IEmitCode;
 *  1. It is the responsibility of the producers of IEmitCode to emit the relevant
 *     jumps for the Lmissing and Lpresent labels (cb.goto or similar)
 *  2. It is the responsibility of consumers to define these labels and to
 *     prevent the pcode from being used on any code path taken as a result of
 *     jumping to Lmissing.
 */
object IEmitCode {
  def apply[A](cb: EmitCodeBuilder, m: Code[Boolean], value: => A): IEmitCodeGen[A] = {
    Code.constBoolValue(m) match {
      case Some(false) => present(cb, value)
      case _ =>
        val Lmissing = CodeLabel()
        val Lpresent = CodeLabel()
        cb.ifx(m, cb.goto(Lmissing))
        val res: A = value
        cb.goto(Lpresent)
        IEmitCodeGen(Lmissing, Lpresent, res, false)
    }
  }

  def apply[A](Lmissing: CodeLabel, Lpresent: CodeLabel, value: A, required: Boolean): IEmitCodeGen[A] =
    IEmitCodeGen(Lmissing, Lpresent, value, false)

  def present[A](cb: EmitCodeBuilder, value: => A): IEmitCodeGen[A] = {
    val Lpresent = CodeLabel()
    val res: A = value
    cb.goto(Lpresent)
    IEmitCodeGen(CodeLabel(), Lpresent, res, true)
  }

  def missing[A](cb: EmitCodeBuilder, defaultValue: A): IEmitCodeGen[A] = {
    val Lmissing = CodeLabel()
    cb.goto(Lmissing)
    IEmitCodeGen(Lmissing, CodeLabel(), defaultValue, false)
  }

  def multiMapEmitCodes(cb: EmitCodeBuilder, seq: IndexedSeq[EmitCode])(f: IndexedSeq[PCode] => PCode): IEmitCode = {
    val Lmissing = CodeLabel()
    val Lpresent = CodeLabel()

    val pcs = seq.map { elem =>
      val iec = elem.toI(cb)

      cb.define(iec.Lmissing)
      cb.goto(Lmissing)
      cb.define(iec.Lpresent)

      iec.value
    }
    val pc = f(pcs)
    cb.goto(Lpresent)

    IEmitCodeGen(Lmissing, Lpresent, pc, seq.forall(_.required))
  }

  def multiFlatMap[A, B, C](seq: IndexedSeq[A], toIec: A => IEmitCodeGen[B], cb: EmitCodeBuilder)
                           (f: IndexedSeq[B] => IEmitCodeGen[C]): IEmitCodeGen[C] = {
    val Lmissing = CodeLabel()

    var required: Boolean = true
    val pcs = seq.map { elem =>
      val iec = toIec(elem)
      required = required && iec.required


      cb.define(iec.Lmissing)
      cb.goto(Lmissing)
      cb.define(iec.Lpresent)

      iec.value
    }
    val iec = f(pcs)
    required = required && iec.required
    cb.define(iec.Lmissing)
    cb.goto(Lmissing)

    IEmitCodeGen(Lmissing, iec.Lpresent, iec.value, required)
  }
}

object IEmitCodeGen {
  implicit class IEmitCode(val iec: IEmitCodeGen[PCode]) extends AnyVal {
    def pc: PCode = iec.value
    def pt: PType = pc.pt

    def memoize(cb: EmitCodeBuilder, name: String): EmitValue =
      cb.memoize(iec, name)
  }
}

case class IEmitCodeGen[+A](Lmissing: CodeLabel, Lpresent: CodeLabel, value: A, val required: Boolean) {

  lazy val emitType: EmitType = {
    value match {
      case pc: PCode => EmitType(pc.st, required)
      case _ => throw new UnsupportedOperationException(s"emitType on $value")
    }
  }

  // This method is a very temporary patch until we can properly separate SCode and PCode
  def typecast[T]: IEmitCodeGen[T] = IEmitCodeGen(Lmissing, Lpresent, value.asInstanceOf[T], required)

  def map[B](cb: EmitCodeBuilder)(f: (A) => B): IEmitCodeGen[B] = {
    val Lpresent2 = CodeLabel()
    cb.define(Lpresent)
    val value2 = f(value)
    cb.goto(Lpresent2)
    IEmitCodeGen(Lmissing, Lpresent2, value2, required)
  }

  def mapMissing(cb: EmitCodeBuilder)(ifMissing: => Unit): IEmitCodeGen[A] = {
    val Lmissing2 = CodeLabel()
    cb.define(Lmissing)
    ifMissing
    cb.goto(Lmissing2)
    IEmitCodeGen(Lmissing2, Lpresent, value, required)
  }

  def flatMap[B](cb: EmitCodeBuilder)(f: (A) => IEmitCodeGen[B]): IEmitCodeGen[B] = {
    cb.define(Lpresent)
    val ec2 = f(value)
    cb.define(ec2.Lmissing)
    cb.goto(Lmissing)
    IEmitCodeGen(Lmissing, ec2.Lpresent, ec2.value, required && ec2.required)
  }

  def handle(cb: EmitCodeBuilder, ifMissing: => Unit): A = {
    cb.define(Lmissing)
    ifMissing
    cb.define(Lpresent)
    value
  }

  def get(cb: EmitCodeBuilder, errorMsg: String = "expected non-missing"): A =
    handle(cb, cb._fatal(errorMsg))

  def consume(cb: EmitCodeBuilder, ifMissing: => Unit, ifPresent: (A) => Unit): Unit = {
    val Lafter = CodeLabel()
    cb.define(Lmissing)
    ifMissing
    if (cb.isOpenEnded) cb.goto(Lafter)
    cb.define(Lpresent)
    ifPresent(value)
    cb.define(Lafter)
  }

  def consumePCode(cb: EmitCodeBuilder, ifMissing: => PCode, ifPresent: (A) => PCode): PCode = {
    val Lafter = CodeLabel()
    cb.define(Lmissing)
    val missingValue = ifMissing
    val pt = missingValue.pt
    val ret = cb.emb.newPLocal(pt)
    cb.assign(ret, missingValue)
    cb.goto(Lafter)
    cb.define(Lpresent)
    val presentValue = ifPresent(value)
    assert(presentValue.pt == pt)
    cb.assign(ret, presentValue)
    cb.define(Lafter)
    ret
  }

  def consumeCode[B: TypeInfo](cb: EmitCodeBuilder, ifMissing: => Code[B], ifPresent: (A) => Code[B]): Code[B] = {
    val ret = cb.emb.newLocal[B]("iec_consumeCode")
    consume(cb, cb.assign(ret, ifMissing), a => cb.assign(ret, ifPresent(a)))
    ret
  }
}

object EmitCode {
  def apply(setup: Code[Unit], m: Code[Boolean], pv: PCode): EmitCode = {
    val mCC = Code(setup, m).toCCode
    val iec = IEmitCode(new CodeLabel(mCC.Ltrue), new CodeLabel(mCC.Lfalse), pv, false)
    val result = new EmitCode(new CodeLabel(mCC.entry), iec)
    result
  }

  def unapply(ec: EmitCode): Option[(Code[Unit], Code[Boolean], PCode)] =
    Some((ec.setup, ec.m, ec.pv))

  def apply(setup: Code[Unit], ec: EmitCode): EmitCode = {
    val Lstart = CodeLabel()
    Code(Lstart, setup, ec.start.goto)
    new EmitCode(Lstart, ec.iec)
  }

  def present(mb: EmitMethodBuilder[_], pc: PCode): EmitCode = EmitCode.fromI(mb)(cb => IEmitCode.present(cb, pc))

  def missing(mb: EmitMethodBuilder[_], pt: PType): EmitCode = EmitCode.fromI(mb)(cb => IEmitCode.missing(cb, pt.defaultValue(mb)))

  def fromI(mb: EmitMethodBuilder[_])(f: (EmitCodeBuilder) => IEmitCode): EmitCode = {
    val cb = EmitCodeBuilder(mb)
    val iec = f(cb)
    val setup = cb.result()
    new EmitCode(new CodeLabel(setup.start), iec)
  }
}

class EmitCode(private val start: CodeLabel, private val iec: IEmitCode) {

  def required: Boolean = iec.required

  lazy val emitType: EmitType = iec.emitType

  def emitParamType: PCodeEmitParamType = PCodeEmitParamType(st.pType)

  def st: SType = iec.value.st

  def pv: PCode = iec.value

  def setup: Code[Unit] = Code._empty

  val m: Code[Boolean] = new CCode(start.L, iec.Lmissing.L, iec.Lpresent.L)

  def pt: PType = pv.pt

  def v: Code[_] = pv.code

  def value[T]: Code[T] = coerce[T](v)

  def toI(cb: EmitCodeBuilder): IEmitCode = {
    cb.goto(start)
    iec
  }

  def castTo(mb: EmitMethodBuilder[_], region: Value[Region], destType: PType, deepCopy: Boolean = false): EmitCode = {
    EmitCode.fromI(mb)(cb => toI(cb).map(cb)(_.castTo(cb, region, destType)))
  }

  def codeTuple(): IndexedSeq[Code[_]] = {
    val tc = pv.codeTuple()
    if (pt.required)
      tc
    else
      tc :+ m
  }

  def missingIf(mb: EmitMethodBuilder[_], cond: Code[Boolean]): EmitCode =
    EmitCode.fromI(mb) { cb =>
      val Ltrue = CodeLabel()
      val Lfalse = CodeLabel()
      cb.ifx(cond, cb.goto(Ltrue), cb.goto(Lfalse))
      cb.define(Lfalse)
      val eci = toI(cb)
      cb.define(Ltrue)
      cb.goto(eci.Lmissing)
      eci
    }

  def get(): PCode =
    PCode(pv.pt, Code(setup, m.orEmpty(Code._fatal[Unit]("expected non-missing")), pv.code))

  def asVoid(): Code[Unit] = {
    require(pv.pt == PVoid)
    Code(setup, Code.toUnit(m))
  }
}

abstract class EmitSettable extends EmitValue {
  def store(cb: EmitCodeBuilder, ec: EmitCode): Unit

  def store(cb: EmitCodeBuilder, iec: IEmitCode): Unit
}

abstract class PresentEmitSettable extends EmitValue {
  def store(cb: EmitCodeBuilder, pc: PCode): Unit
}

class RichIndexedSeqEmitSettable(is: IndexedSeq[EmitSettable]) {
  def load(): IndexedSeq[EmitCode] = is.map(_.load)
}

object LoopRef {
  def apply(cb: EmitCodeBuilder, L: CodeLabel, args: IndexedSeq[(String, PType)], pool: Value[RegionPool]): LoopRef = {
    val (loopArgs, tmpLoopArgs) = args.zipWithIndex.map { case ((name, pt), i) =>
      (cb.emb.newEmitField(s"$name$i", pt), cb.emb.newEmitField(s"tmp$name$i", pt))
    }.unzip

    val r1: Settable[Region] = cb.newLocal[Region]("loop_ref_r1")
    cb.assign(r1, Region.stagedCreate(Region.REGULAR, pool))

    val r2: Settable[Region] = cb.newLocal[Region]("loop_ref_r2")
    cb.assign(r2, Region.stagedCreate(Region.REGULAR, pool))

    LoopRef(L, args.map(_._2), loopArgs, tmpLoopArgs, r1, r2)
  }
}

case class LoopRef(
  L: CodeLabel,
  loopTypes: IndexedSeq[PType],
  loopArgs: IndexedSeq[EmitSettable],
  tmpLoopArgs: IndexedSeq[EmitSettable],
  r1: Settable[Region],
  r2: Settable[Region])

abstract class EstimableEmitter[C] {
  def emit(mb: EmitMethodBuilder[C]): Code[Unit]

  def estimatedSize: Int
}

class Emit[C](
  val ctx: EmitContext,
  val cb: EmitClassBuilder[C]) { emitSelf =>

  val methods: mutable.Map[(String, Seq[Type], Seq[PType], PType), EmitMethodBuilder[C]] = mutable.Map()


  private[ir] def emitVoid(cb: EmitCodeBuilder, ir: IR, region: Value[Region], env: E, container: Option[AggContainer], loopEnv: Option[Env[LoopRef]]): Unit = {
    val mb: EmitMethodBuilder[C] = cb.emb.asInstanceOf[EmitMethodBuilder[C]]

    def emit(ir: IR, mb: EmitMethodBuilder[C] = mb, region: Value[Region] = region, env: E = env, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): EmitCode =
      this.emit(ir, mb, region, env, container, loopEnv)

    def emitStream(ir: IR, outerRegion: Value[Region], mb: EmitMethodBuilder[C] = mb): EmitCode =
      EmitCode.fromI(mb)(cb => EmitStream.produce(this, ir, cb, outerRegion, env, container))

    def emitVoid(ir: IR, cb: EmitCodeBuilder = cb, region: Value[Region] = region, env: E = env, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): Unit =
      this.emitVoid(cb, ir, region, env, container, loopEnv)

    def emitI(ir: IR, region: Value[Region] = region, env: E = env, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): IEmitCode =
      this.emitI(ir, cb, region, env, container, loopEnv)

    (ir: @unchecked) match {
      case Void() =>
        Code._empty

      case Begin(xs) =>
        xs.foreach(x => emitVoid(x))

      case If(cond, cnsq, altr) =>
        assert(cnsq.typ == TVoid && altr.typ == TVoid)

        emitI(cond).consume(cb, {}, m => cb.ifx(m.asBoolean.boolCode(cb), emitVoid(cnsq), emitVoid(altr)))

      case Let(name, value, body) =>
        val xVal = if (value.typ.isInstanceOf[TStream]) emitStream(value, region) else emit(value)
        cb.withScopedMaybeStreamValue(xVal, s"let_$name") { ev =>
          emitVoid(body, env = env.bind(name, ev))
        }

      case StreamFor(a, valueName, body) =>
        val streamType = coerce[PStream](a.pType)

        emitStream(a, region).toI(cb).consume(cb,
          {},
          { case stream: SStreamCode =>
            val producer = stream.producer
            producer.memoryManagedConsume(region, cb) { cb =>
              cb.withScopedMaybeStreamValue(producer.element, s"streamfor_$valueName") { ev =>
                emitVoid(body, region = producer.elementRegion, env = env.bind(valueName -> ev))
              }
            }
          })

      case x@InitOp(i, args, sig) =>
        val AggContainer(aggs, sc, _) = container.get
        assert(aggs(i) == sig.state)
        val rvAgg = agg.Extract.getAgg(sig)

        val argVars = args
          .map { a => emit(a, container = container.flatMap(_.nested(i, init = true))) }
          .toArray

        sc.newState(cb, i)
        rvAgg.initOp(cb, sc.states(i), argVars)

      case x@SeqOp(i, args, sig) =>
        val AggContainer(aggs, sc, _) = container.get
        assert(sig.state == aggs(i))
        val rvAgg = agg.Extract.getAgg(sig)

        val argVars = args
          .map { a => emit(a, container = container.flatMap(_.nested(i, init = false))) }
          .toArray

        rvAgg.seqOp(cb, sc.states(i), argVars)

      case x@CombOp(i1, i2, sig) =>
        val AggContainer(aggs, sc, _) = container.get
        assert(sig.state == aggs(i1) && sig.state == aggs(i2))
        val rvAgg = agg.Extract.getAgg(sig)
        rvAgg.combOp(cb, sc.states(i1), sc.states(i2))

      case x@SerializeAggs(start, sIdx, spec, sigs) =>
        val AggContainer(_, sc, _) = container.get
        val ob = mb.genFieldThisRef[OutputBuffer]()
        val baos = mb.genFieldThisRef[ByteArrayOutputStream]()

        cb.assign(baos, Code.newInstance[ByteArrayOutputStream]())
        cb.assign(ob, spec.buildCodeOutputBuffer(baos))

        Array.range(start, start + sigs.length)
          .foreach { idx =>
            sc.states(idx).serialize(spec)(cb, ob)
          }

        cb += ob.invoke[Unit]("flush")
        cb += ob.invoke[Unit]("close")
        cb += mb.setSerializedAgg(sIdx, baos.invoke[Array[Byte]]("toByteArray"))

      case DeserializeAggs(start, sIdx, spec, sigs) =>
        val AggContainer(_, sc, _) = container.get
        val ib = mb.genFieldThisRef[InputBuffer]()

        val ns = sigs.length
        val deserializers = sc.states.states
          .slice(start, start + ns)
          .map(sc => sc.deserialize(BufferSpec.defaultUncompressed))

        Array.range(start, start + ns).foreach(i => sc.newState(cb, i))

        cb.assign(ib, spec.buildCodeInputBuffer(
            Code.newInstance[ByteArrayInputStream, Array[Byte]](
              mb.getSerializedAgg(sIdx))))
        cb += mb.freeSerializedAgg(sIdx)

        (0 until ns).foreach { j =>
          deserializers(j)(cb, ib)
        }

        cb.assign(ib, Code._null)

      case Die(m, typ, errorId) =>
        val cm = emitI(m)
        val msg = cm.consumeCode(cb, "<exception message missing>", _.asString.loadString())
        cb._throw(Code.newInstance[HailException, String, Int](msg, errorId))

      case x@WriteMetadata(annotations, writer) =>
        writer.writeMetadata(emitI(annotations), cb, region)

      case CombOpValue(i, value, aggSig) =>
        val AggContainer(_, sc, _) = container.get
        val rvAgg = agg.Extract.getAgg(aggSig)
        val tempState = AggStateSig.getState(aggSig.state, mb.ecb)
        val aggStateOffset = mb.genFieldThisRef[Long](s"combOpValue_${i}_state");

        val v = emitI(value)
        v.consume(cb,
          cb._fatal("cannot combOp a missing value"),
          { serializedValue =>
            cb.assign(aggStateOffset, region.allocate(tempState.storageType.alignment, tempState.storageType.byteSize))
            tempState.createState(cb)
            tempState.newState(cb)
            tempState.deserializeFromBytes(cb, serializedValue.pt.asInstanceOf[PBinary], serializedValue.code.asInstanceOf[Code[Long]])
            rvAgg.combOp(cb, sc.states(i), tempState)
          }
        )

      case InitFromSerializedValue(i, value, sig) =>
        val AggContainer(aggs, sc, _) = container.get
        assert(aggs(i) == sig)

        val v = emitI(value)
        v.consume(cb,
          cb._fatal("cannot initialize aggs from a missing value"),
          { serializedValue =>
            sc.states(i).createState(cb)
            sc.newState(cb, i)
            sc.states(i).deserializeFromBytes(cb, serializedValue.pt.asInstanceOf[PBinary], serializedValue.code.asInstanceOf[Code[Long]])
          }
        )
    }
  }

  private[ir] def emitI(ir: IR, cb: EmitCodeBuilder, env: E, container: Option[AggContainer]): IEmitCode = {
    val region = cb.emb.getCodeParam[Region](1)
    emitI(ir, cb, region, env, container, None)
  }

  private[ir] def emitI(ir: IR, cb: EmitCodeBuilder, region: Value[Region], env: E,
    container: Option[AggContainer], loopEnv: Option[Env[LoopRef]]
  ): IEmitCode = {
    val mb: EmitMethodBuilder[C] = cb.emb.asInstanceOf[EmitMethodBuilder[C]]

    def emitI(ir: IR, region: Value[Region] = region, env: E = env, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): IEmitCode =
      this.emitI(ir, cb, region, env, container, loopEnv)

    def emitInNewBuilder(cb: EmitCodeBuilder, ir: IR, region: Value[Region] = region, env: E = env, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): IEmitCode =
      this.emitI(ir, cb, region, env, container, loopEnv)

    def emitInMethod(cb: EmitCodeBuilder, ir: IR): IEmitCode =
      this.emitI(ir, cb, Env.empty, container)

    def emitStream(ir: IR, cb: EmitCodeBuilder, outerRegion: Value[Region] ): IEmitCode =
      EmitStream.produce(this, ir, cb, outerRegion, env, container)

    def emitVoid(ir: IR, env: E = env, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): Unit =
      this.emitVoid(cb, ir: IR, region, env, container, loopEnv)

    def emitFallback(ir: IR, env: E = env, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): IEmitCode =
      this.emit(ir, mb, region, env, container, loopEnv, fallingBackFromEmitI = true).toI(cb)

    def emitDeforestedNDArrayI(ir: IR): IEmitCode =
      deforestNDArrayI(ir, cb, region, env)

    def emitNDArrayColumnMajorStrides(ir: IR): IEmitCode = {
      emitI(ir).map(cb){case pNDCode: PNDArrayCode =>
        val pNDValue = pNDCode.memoize(cb, "ndarray_column_major_check")
        LinalgCodeUtils.checkColMajorAndCopyIfNeeded(pNDValue, cb, region)
      }
    }

    val pt = ir.pType

    if (pt == PVoid) {
      emitVoid(ir)
      return IEmitCode(CodeLabel(), CodeLabel(), PCode._empty, required = true)
    }

    def presentPC(pc: PCode): IEmitCode = IEmitCode.present(cb, pc)

    def presentC(c: Code[_]): IEmitCode = presentPC(PCode(pt, c))

    val result: IEmitCode = (ir: @unchecked) match {
      case I32(x) =>
        presentC(const(x))
      case I64(x) =>
        presentC(const(x))
      case F32(x) =>
        presentC(const(x))
      case F64(x) =>
        presentC(const(x))
      case s@Str(x) =>
        presentPC(mb.addLiteral(x, coerce[PString](s.pType)))
      case x@UUID4(_) =>
        presentPC(PCode(x.pType, coerce[PString](x.pType).
          allocateAndStoreString(mb, region, Code.invokeScalaObject0[String](
            Class.forName("is.hail.expr.ir.package$"), "uuid4"))))
      case x@Literal(t, v) =>
        presentPC(mb.addLiteral(v, x.pType))
      case x@EncodedLiteral(codec, value) =>
        assert(x.pType == codec.decodedPType())
        presentPC(mb.addEncodedLiteral(x))
      case True() =>
        presentC(const(true))
      case False() =>
        presentC(const(false))
      case Consume(value) =>
        emitI(value).map(cb){pc =>
          cb.memoizeField(pc, "consumed_field")
          // Ignore pc, just return a 1
          PCode(ir.pType, 1L)
        }
      case Cast(v, typ) =>
        val iec = emitI(v)
        val cast = Casts.get(v.typ, typ)
        iec.map(cb)(pc => PCode(pt, cast(pc.code)))
      case CastRename(v, _typ) =>
        emitI(v)
          .map(cb)(pc => PCode(pt, pc.code))
      case NA(typ) =>
        IEmitCode(cb, const(true), pt.defaultValue(cb.emb))
      case IsNA(v) =>
        val m = emitI(v).consumeCode(cb, true, _ => false)
        presentC(m)

      case Coalesce(values) =>
        val coalescedValue = mb.newPLocal("coalesce_value", pt)

        val emittedValues = values.map(v => EmitCode.fromI(cb.emb)(cb => emitInNewBuilder(cb, v)))
        val Ldefined = CodeLabel()
        val Lmissing = CodeLabel()

        emittedValues.foreach { value =>
          value.toI(cb).consume(cb,
            {}, // fall through to next check
            { sc =>
              cb.assign(coalescedValue, sc.castTo(cb, region, pt))
              cb.goto(Ldefined)
            })
        }

        cb.goto(Lmissing)

        IEmitCode(Lmissing, Ldefined, coalescedValue.load(), emittedValues.forall(_.required))

      case If(cond, cnsq, altr) =>
        assert(cnsq.typ == altr.typ)

        emitI(cond).flatMap(cb) { condValue =>

          val codeCnsq = EmitCode.fromI(cb.emb)(cb => emitInNewBuilder(cb, cnsq))
          val codeAltr = EmitCode.fromI(cb.emb)(cb => emitInNewBuilder(cb, altr))

          val Lmissing = CodeLabel()
          val Ldefined = CodeLabel()
          val out = mb.newPLocal(pt)
          cb.ifx(condValue.asBoolean.boolCode(cb), {
            codeCnsq.toI(cb).consume(cb,
              {
                cb.goto(Lmissing)
              }, {sc =>
                cb.assign(out, sc.castTo(cb, region, pt))
              })
          }, {
            codeAltr.toI(cb).consume(cb,
              {
                cb.goto(Lmissing)
              }, {sc =>
                cb.assign(out, sc.castTo(cb, region, pt))
              })
          })
          cb.goto(Ldefined)

          IEmitCode(Lmissing, Ldefined, out.load(), codeCnsq.required && codeAltr.required)
        }

      case x@MakeStruct(fields) =>
        val scode = x.pType.asInstanceOf[PCanonicalBaseStruct].constructFromFields(cb,
          region,
          fields.map { case (name, x) =>
            EmitCode.fromI(cb.emb)(cb => emitInNewBuilder(cb, x))
          }.toFastIndexedSeq,
          deepCopy = false)
        presentPC(scode)

      case x@MakeTuple(fields) =>
        val scode = x.pType.asInstanceOf[PCanonicalBaseStruct].constructFromFields(cb,
          region,
          fields.map { case (_, x) =>
            EmitCode.fromI(cb.emb)(emitInNewBuilder(_, x))
          }.toFastIndexedSeq,
          deepCopy = false)
        presentPC(scode)

      case x@SelectFields(oldStruct, fields) =>
        emitI(oldStruct)
          .map(cb) { case sc: SBaseStructCode =>
            val sv = sc.memoize(cb, "select_fields_scode")
            x.pType.asInstanceOf[PCanonicalBaseStruct].constructFromFields(cb,
              region,
              fields.map { field =>
                EmitCode.fromI(cb.emb)(cb => sv.loadField(cb, field).typecast[PCode])
              }.toFastIndexedSeq,
              deepCopy = false)
          }

      case x@InsertFields(old, fields, fieldOrder) =>
        if (fields.isEmpty)
          emitI(old)
        else {
          val codeOld = emitI(old)
          val updateMap = Map(fields: _*)

          codeOld.map(cb) { oldPC =>
            val oldPV = oldPC.asBaseStruct.memoize(cb, "insert_fields_old")

            val itemsEC = x.pType.fields.map { f =>
              updateMap.get(f.name) match {
                case Some(vir) =>
                  EmitCode.fromI(mb)(emitInNewBuilder(_, vir))
                case None =>
                  EmitCode.fromI(mb)(oldPV.loadField(_, f.name).typecast[PCode])
              }
            }

            x.pType.asInstanceOf[PCanonicalBaseStruct]
             .constructFromFields(cb, region, itemsEC, deepCopy = false)
             .asPCode
          }
        }

      case ApplyBinaryPrimOp(op, l, r) =>
        emitI(l).flatMap(cb) { pcL =>
          emitI(r).map(cb)(pcR => PCode(pt, BinaryOp.emit(op, l.typ, r.typ, pcL.code, pcR.code)))
        }
      case ApplyUnaryPrimOp(op, x) =>
        emitI(x).map(cb)(pc => PCode(pt, UnaryOp.emit(op, x.typ, pc.code)))
      case ApplyComparisonOp(op, l, r) =>
        if (op.strict) {
          emitI(l).flatMap(cb) { l =>
            emitI(r).map(cb) { r =>
              val f = op.codeOrdering(cb.emb.ecb, l.st, r.st)
              PCode(pt, f(cb, EmitCode.present(cb.emb, l), EmitCode.present(cb.emb, r)))
            }
          }
        } else {
          val lc = emitI(l).memoize(cb, "l")
          val rc = emitI(r).memoize(cb, "r")
          val f = op.codeOrdering(cb.emb.ecb, lc.st, rc.st)
          presentC(f(cb, lc, rc))
        }

      case x@MakeArray(args, _) =>
        val pType = x.pType.asInstanceOf[PCanonicalArray]

        val (pushElement, finish) = pType.constructFromFunctions(cb, region, args.size, deepCopy = false)
        for (arg <- args) {
          val v = emitI(arg)
          pushElement(cb, v)
        }
        presentPC(finish(cb))

      case ArrayZeros(length) =>
        emitI(length).map(cb) { case n: SInt32Code =>
          val outputPType = coerce[PArray](ir.pType)
          val elementSize = outputPType.elementByteSize
          val numElements = cb.newLocal[Int]("n_elements", n.intCode(cb))
          val arrayAddress = cb.newLocal[Long]("array_addr", outputPType.allocate(region, numElements))
          cb += outputPType.stagedInitialize(arrayAddress, numElements)
          cb += Region.setMemory(outputPType.firstElementOffset(arrayAddress), numElements.toL * elementSize, 0.toByte)
          PCode(pt, arrayAddress)
        }

      case x@ArrayRef(a, i, s) =>
        def boundsCheck(cb: EmitCodeBuilder, index: Value[Int], len: Value[Int]): Unit = {
          s match {
            case Str(constant) =>
              val baseMsg = constant match {
                case "" => s"\n----------\nIR:\n${ Pretty.short(x) }"
                case c => s"\n----------\nPython traceback:\n$c"
              }
              val bcMb = mb.getOrGenEmitMethod("arrayref_bounds_check", ("arrayref_bounds_check", baseMsg),
                IndexedSeq[ParamType](IntInfo, IntInfo), UnitInfo)({ mb =>
                mb.voidWithBuilder { cb =>
                  val index = mb.getCodeParam[Int](1)
                  val len = mb.getCodeParam[Int](2)
                  cb.ifx(index < 0 || index >= len, {
                    cb._fatal(const("array index out of bounds: index=")
                      .concat(index.toS)
                      .concat(", length=")
                      .concat(len.toS)
                      .concat(baseMsg))
                  })

                }
              })
              cb.invokeVoid(bcMb, index, len)
            case s =>
              cb.ifx(index < 0 || index >= len, {
                val msg = cb.newLocal[String]("arrayref_msg", const("array index out of bounds: index=")
                  .concat(index.toS)
                  .concat(", length=")
                  .concat(len.toS))
                emitI(s).consume(cb, (), sc => cb.assign(msg, msg.concat(sc.asString.loadString())))
                cb._fatal(msg)
              })
          }
        }

        emitI(a).flatMap(cb) { (ac) =>
          emitI(i).flatMap(cb) { (ic) =>
            val av = ac.asIndexable.memoize(cb, "aref_a")
            val iv = cb.newLocal("i", ic.asInt.intCode(cb))
            boundsCheck(cb, iv, av.loadLength())
            av.loadElement(cb, iv).typecast[PCode]
          }
        }

      case ArrayLen(a) =>
        emitI(a).map(cb) { (ac) =>
          PCode(pt, ac.asIndexable.loadLength())
        }

      case GetField(o, name) =>
        emitI(o).flatMap(cb) { oc =>
          val ov = oc.asBaseStruct.memoize(cb, "get_tup_elem_o")
          ov.loadField(cb, name).typecast[PCode]
        }

      case GetTupleElement(o, i) =>
        emitI(o).flatMap(cb) { oc =>
          val ov = oc.asBaseStruct.memoize(cb, "get_tup_elem_o")
          ov.loadField(cb, oc.pt.asInstanceOf[PTuple].fieldIndex(i)).typecast[PCode]
        }

      case x@LowerBoundOnOrderedCollection(orderedCollection, elem, onKey) =>
        emitI(orderedCollection).map(cb) { a =>
          val typ: PContainer = coerce[PIterable](a.pt).asPContainer
          val e = EmitCode.fromI(cb.emb)(cb => this.emitI(elem, cb, region, env, container, loopEnv))
          val bs = new BinarySearch[C](mb, typ, e.pt, keyOnly = onKey)
          val arr = SingleCodePCode.fromPCode(cb, a, region)
          PCode(pt, bs.getClosestIndex(arr.code.asInstanceOf[Code[Long]], e.m, e.v))
        }

      case GroupByKey(collection) =>
        // sort collection by group
        val collectionTyp = coerce[PStream](collection.pType)
        val keyValTyp = coerce[PBaseStruct](collectionTyp.elementType)
        val keyTyp = keyValTyp.types(0)
        val valTyp = keyValTyp.types(1)
        val dictTyp = coerce[PCanonicalDict](ir.pType)
        val groupTyp = dictTyp.elementType
        val arrayTyp = PCanonicalArray(groupTyp, required = true)

        val sortedElts = new StagedArrayBuilder(keyValTyp, mb, 16)
        val sorter = new ArraySorter(EmitRegion(mb, region), sortedElts)

        val (k1, k2) = keyValTyp match {
          case t: PStruct => GetField(In(0, PCodeEmitParamType(t)), "key") -> GetField(In(1, PCodeEmitParamType(t)), "key")
          case t: PTuple =>
            assert(t.fields(0).index == 0)
            GetTupleElement(In(0, PCodeEmitParamType(t)), 0) -> GetTupleElement(In(1, PCodeEmitParamType(t)), 0)
        }

        val compare = ApplyComparisonOp(Compare(keyValTyp.types(0).virtualType), k1, k2) < 0
        InferPType(compare)
        val leftRightComparatorNames = Array.empty[String]
        val sortF = sortedElts.ti match {
          case BooleanInfo => makeDependentSortingFunction[Boolean](region, keyValTyp, compare, env, leftRightComparatorNames)
          case IntInfo => makeDependentSortingFunction[Int](region, keyValTyp, compare, env, leftRightComparatorNames)
          case LongInfo => makeDependentSortingFunction[Long](region, keyValTyp, compare, env, leftRightComparatorNames)
          case FloatInfo => makeDependentSortingFunction[Float](region, keyValTyp, compare, env, leftRightComparatorNames)
          case DoubleInfo => makeDependentSortingFunction[Double](region, keyValTyp, compare, env, leftRightComparatorNames)
        }

        val groupSizes = new StagedArrayBuilder(PInt32(), mb, 0)

        val (lastKey, currKey) = (keyValTyp.virtualType: @unchecked) match {
          case ts: TStruct =>
            GetField(In(0, PCodeEmitParamType(keyValTyp)), ts.fieldNames(0)) -> GetField(In(1, PCodeEmitParamType(keyValTyp)), ts.fieldNames(0))
          case tt: TTuple =>
            GetTupleElement(In(0, PCodeEmitParamType(keyValTyp)), tt.fields(0).index) -> GetTupleElement(In(1, PCodeEmitParamType(keyValTyp)), tt.fields(0).index)
        }
        val compare2 = ApplyComparisonOp(EQWithNA(keyTyp.virtualType), lastKey, currKey)
        InferPType(compare2)
        val isSame = mb.genEmitMethod("isSame",
          FastIndexedSeq(typeInfo[Region], PCodeEmitParamType(keyValTyp), PCodeEmitParamType(keyValTyp)),
          BooleanInfo)
        isSame.emitWithBuilder { cb =>
          emitInMethod(cb, compare2).consumeCode[Boolean](cb, true, _.asBoolean.boolCode(cb))
        }

        val eltIdx = mb.newLocal[Int]("groupByKey_eltIdx")
        val grpIdx = mb.newLocal[Int]("groupByKey_grpIdx")
        val withinGrpIdx = mb.newLocal[Int]("groupByKey_withinGrpIdx")
        val outerSize = mb.newLocal[Int]("groupByKey_outerSize")
        val groupSize = mb.newLocal[Int]("groupByKey_groupSize")

        emitStream(collection, cb, region).map(cb) { case stream: SStreamCode =>

          StreamUtils.writeToArrayBuilder(cb, stream.producer, sortedElts, region)
          cb += sorter.sort(sortF)
          cb += sorter.pruneMissing
          cb += groupSizes.clear
          cb.assign(eltIdx, 0)
          cb.assign(groupSize, 0)

          cb.whileLoop(eltIdx < sortedElts.size, {
            val bottomOfLoop = CodeLabel()
            val newGroup = CodeLabel()

            cb.assign(groupSize, groupSize + 1)
            cb.ifx(eltIdx.ceq(sortedElts.size - 1), {
              cb.goto(newGroup)
            }, {
              cb.ifx(cb.invokeCode[Boolean](isSame, region, sortedElts.applyEV(mb, eltIdx), sortedElts.applyEV(mb, eltIdx + 1)), {
                cb.goto(bottomOfLoop)
              }, {
                cb.goto(newGroup)
              })
            })
            cb.define(newGroup)
            cb += groupSizes.add(groupSize)
            cb.assign(groupSize, 0)

            cb.define(bottomOfLoop)
            cb.assign(eltIdx, eltIdx + 1)
          })

          cb.assign(outerSize, groupSizes.size)
          val (addGroup, finishOuter) = arrayTyp.constructFromFunctions(cb, region, outerSize, deepCopy = false)

          cb.assign(eltIdx, 0)
          cb.assign(grpIdx, 0)

          cb.whileLoop(grpIdx < outerSize, {
            cb.assign(groupSize, coerce[Int](groupSizes(grpIdx)))
            cb.assign(withinGrpIdx, 0)
            val firstStruct = sortedElts.applyEV(mb, eltIdx).get(cb).asBaseStruct.memoize(cb, "GroupByKey_firstStruct")
            val key = EmitCode.fromI(mb) { cb => firstStruct.loadField(cb, 0).typecast[PCode] }
            val group = EmitCode.fromI(mb) { cb =>
              val (addElt, finishInner) = PCanonicalArray(valTyp, required = true)
                .constructFromFunctions(cb, region, groupSize, deepCopy = false)
              cb.whileLoop(withinGrpIdx < groupSize, {
                val struct = sortedElts.applyEV(mb, eltIdx).get(cb).asBaseStruct.memoize(cb, "GroupByKey_struct")
                addElt(cb, struct.loadField(cb, 1).typecast[PCode])
                cb.assign(eltIdx, eltIdx + 1)
                cb.assign(withinGrpIdx, withinGrpIdx + 1)
              })
              IEmitCode.present(cb, finishInner(cb))
            }
            val elt = groupTyp.constructFromFields(cb, region, FastIndexedSeq(key, group), deepCopy = false)
            addGroup(cb, IEmitCode.present(cb, elt))
            cb.assign(grpIdx, grpIdx + 1)
          })

          dictTyp.construct(finishOuter(cb))
        }

      case x@StreamLen(a) =>
        emitStream(a, cb, region).map(cb) { case stream: SStreamCode =>
          val producer = stream.producer
          producer.length match {
            case Some(len) =>
              producer.initialize(cb)
              val xLen = cb.newLocal[Int]("streamlen_x", len)
              producer.close(cb)
              PCode(x.pType, xLen)
            case None =>
              val count = cb.newLocal[Int]("stream_length", 0)
              producer.memoryManagedConsume(region, cb) { cb =>
                producer.element.toI(cb).consume(cb, {}, _ => {})
                cb.assign(count, count + 1)
              }
              producer.element.pv match {
                case SStreamCode(_, nested) => StreamProducer.defineUnusedLabels(nested, mb)
                case _ =>
              }
              PCode(x.pType, count)
          }
        }

      case x@MakeNDArray(dataIR, shapeIR, rowMajorIR, errorId) =>
        val xP = coerce[PCanonicalNDArray](x.pType)
        val shapePType = coerce[PTuple](shapeIR.pType)
        val nDims = shapePType.size

        emitI(rowMajorIR).flatMap(cb) { isRowMajorCode =>
          emitI(shapeIR).flatMap(cb) { case shapeTupleCode: PBaseStructCode =>
            emitI(dataIR).map(cb) { case dataCode: PIndexableCode =>
              val shapeTupleValue = shapeTupleCode.memoize(cb, "make_ndarray_shape")
              val memoData = dataCode.memoize(cb, "make_nd_array_memoized_data")

              cb.ifx(memoData.hasMissingValues(cb), {
                cb._throw(Code.newInstance[HailException, String, Int](
                    "Cannot construct an ndarray with missing values.", errorId
              ))})

              (0 until nDims).foreach { index =>
                cb.ifx(shapeTupleValue.isFieldMissing(index),
                  cb.append(Code._fatal[Unit](s"shape missing at index $index")))
              }

              val stridesSettables = (0 until nDims).map(i => cb.newLocal[Long](s"make_ndarray_stride_$i"))

              val shapeValues = (0 until nDims).map { i =>
                shapeTupleValue.loadField(cb, i).get(cb).memoize(cb, s"make_ndarray_shape_${i}").asPValue.value.asInstanceOf[Value[Long]]
              }

              cb.ifx(isRowMajorCode.asBoolean.boolCode(cb), {
                val strides = xP.makeRowMajorStrides(shapeValues, region, cb)

                stridesSettables.zip(strides).foreach { case (settable, stride) =>
                  cb.assign(settable, stride)
                }
              }, {
                val strides = xP.makeColumnMajorStrides(shapeValues, region, cb)
                stridesSettables.zip(strides).foreach { case (settable, stride) =>
                  cb.assign(settable, stride)
                }
              })

              xP.constructByCopyingArray(shapeValues, stridesSettables, memoData.pc.asIndexable, cb, region)
            }
          }
        }
      case NDArrayShape(ndIR) =>
        emitI(ndIR).map(cb){ case pc: PNDArrayCode => pc.shape(cb).asPCode}
      case x@NDArrayReindex(child, indexMap) =>
        val childEC = emitI(child)
        val childPType = coerce[PCanonicalNDArray](child.pType)
        childEC.map(cb){ case pndCode: PNDArrayCode =>
          val pndVal = pndCode.memoize(cb, "ndarray_reindex_child")
          val childShape = pndVal.shapes(cb)
          val childStrides = pndVal.strides(cb)

          val pndAddr = SingleCodePCode.fromPCode(cb, pndVal, region)
          val dataArray = childPType.dataType.loadCheapPCode(cb, childPType.dataPArrayPointer(pndAddr.code.asInstanceOf[Code[Long]]))

          val newShape = indexMap.map { childIndex =>
            if (childIndex < childPType.nDims) childShape(childIndex) else const(1L)
          }
          val newStrides = indexMap.map { childIndex =>
            if (childIndex < childPType.nDims) childStrides(childIndex) else const(0L)
          }

          x.pType.constructByCopyingArray(
            newShape,
            newStrides,
            dataArray,
            cb,
            region)
        }

      case NDArrayRef(nd, idxs, errorId) =>
        val ndt = emitI(nd)

        ndt.flatMap(cb) { case ndCode: PNDArrayCode =>
          val indexEmitCodes = idxs.map(idx => EmitCode.fromI(cb.emb)(emitInNewBuilder(_, idx)))
          IEmitCode.multiMapEmitCodes(cb, indexEmitCodes) { idxPCodes: IndexedSeq[PCode] =>
            val memoizedIndices = idxPCodes.zipWithIndex.map { case (pc, idx) =>
              pc.memoize(cb,s"ref_idx_$idx")
            }

            val ndValue = ndCode.memoize(cb, "reffed_ndarray")
            val idxValues = memoizedIndices.map(_.value.asInstanceOf[Value[Long]])
            cb.append(ndValue.assertInBounds(idxValues, cb, errorId))

            ndValue.loadElement(idxValues, cb).asPCode
          }
        }

      case NDArrayMatMul(lChild, rChild) =>
        emitNDArrayColumnMajorStrides(lChild).flatMap(cb) { case leftPCode: PNDArrayCode =>
          emitNDArrayColumnMajorStrides(rChild).map(cb) { case rightPCode: PNDArrayCode =>
            val lPType = leftPCode.pt
            val rPType = rightPCode.pt

            val leftPVal = leftPCode.memoize(cb, "left_ndarray_matmul")
            val rightPVal = rightPCode.memoize(cb, "right_ndarray_matmul")

            val lShape = leftPVal.shapes(cb)
            val rShape = rightPVal.shapes(cb)

            val unifiedShape = NDArrayEmitter.matmulShape(cb, lShape, rShape)

            val leftBroadcastMask = if (lPType.nDims > 2) NDArrayEmitter.broadcastMask(lShape) else IndexedSeq[Value[Long]]()
            val rightBroadcastMask = if (rPType.nDims > 2) NDArrayEmitter.broadcastMask(rShape) else IndexedSeq[Value[Long]]()

            val outputPType = PCanonicalNDArray(lPType.elementType, TNDArray.matMulNDims(lPType.nDims, rPType.nDims), pt.required)

            if ((lPType.elementType.isInstanceOf[PFloat64] || lPType.elementType.isInstanceOf[PFloat32]) && lPType.nDims == 2 && rPType.nDims == 2) {
              val leftPValAddr = SingleCodePCode.fromPCode(cb, leftPVal, region)
              val rightPValAddr = SingleCodePCode.fromPCode(cb, rightPVal, region)
              val leftDataAddress = lPType.dataFirstElementPointer(leftPValAddr.code.asInstanceOf[Code[Long]])
              val rightDataAddress = rPType.dataFirstElementPointer(rightPValAddr.code.asInstanceOf[Code[Long]])

              val M = lShape(lPType.nDims - 2)
              val N = rShape(rPType.nDims - 1)
              val K = lShape(lPType.nDims - 1)

              val LDA = M
              val LDB = K
              val LDC = M

              val (answerFirstElementAddr, answerFinisher) = outputPType.constructDataFunction(
                IndexedSeq(M, N),
                outputPType.makeColumnMajorStrides(IndexedSeq(M, N), region, cb),
                cb,
                region)

              cb.ifx((M cne 0L) && (N cne 0L) && (K cne 0L), {
                cb.append(lPType.elementType match {
                  case PFloat32(_) =>
                    Code.invokeScalaObject13[String, String, Int, Int, Int, Float, Long, Int, Long, Int, Float, Long, Int, Unit](BLAS.getClass, method = "sgemm",
                      "N",
                      "N",
                      M.toI,
                      N.toI,
                      K.toI,
                      1.0f,
                      leftDataAddress,
                      LDA.toI,
                      rightDataAddress,
                      LDB.toI,
                      0.0f,
                      answerFirstElementAddr,
                      LDC.toI
                    )
                  case PFloat64(_) =>
                    Code.invokeScalaObject13[String, String, Int, Int, Int, Double, Long, Int, Long, Int, Double, Long, Int, Unit](BLAS.getClass, method = "dgemm",
                      "N",
                      "N",
                      M.toI,
                      N.toI,
                      K.toI,
                      1.0,
                      leftDataAddress,
                      LDA.toI,
                      rightDataAddress,
                      LDB.toI,
                      0.0,
                      answerFirstElementAddr,
                      LDC.toI
                    )
                })
              },
                { // Fill with zeroes
                  cb.append(Region.setMemory(answerFirstElementAddr, (M * N) * outputPType.elementType.byteSize, 0.toByte))
                }
              )

              answerFinisher(cb)
            } else {
              val numericElementType = coerce[PNumeric](lPType.elementType)
              val eVti = typeToTypeInfo(numericElementType)

              val emitter = new NDArrayEmitter(unifiedShape) {
                override def outputElement(cb: EmitCodeBuilder, idxVars: IndexedSeq[Value[Long]]): PCode = {
                  val element = coerce[Any](cb.newField("matmul_element")(eVti))
                  val k = cb.newField[Long]("ndarray_matmul_k")

                  val (lIndices: IndexedSeq[Value[Long]], rIndices: IndexedSeq[Value[Long]]) = (lPType.nDims, rPType.nDims, idxVars) match {
                    case (1, 1, Seq()) => (IndexedSeq(k), IndexedSeq(k))
                    case (1, _, stack :+ m) =>
                      val rStackVars = NDArrayEmitter.zeroBroadcastedDims(stack, rightBroadcastMask)
                      (IndexedSeq(k), rStackVars :+ k :+ m)
                    case (_, 1, stack :+ n) =>
                      val lStackVars = NDArrayEmitter.zeroBroadcastedDims(stack, leftBroadcastMask)
                      (lStackVars :+ n :+ k, FastIndexedSeq(k))
                    case (_, _, stack :+ n :+ m) =>
                      val lStackVars = NDArrayEmitter.zeroBroadcastedDims(stack, leftBroadcastMask)
                      val rStackVars = NDArrayEmitter.zeroBroadcastedDims(stack, rightBroadcastMask)
                      (lStackVars :+ n :+ k, rStackVars :+ k :+ m)
                  }

                  val kLen = cb.newField[Long]("ndarray_matmul_kLen")

                  def multiply(l: PCode, r: PCode): Code[_] = {
                    (l.st, r.st) match {
                      case (_: SInt32, _: SInt32) =>
                        l.asInt.intCode(cb) * r.asInt.intCode(cb)
                      case (_: SInt64, _: SInt64) =>
                        l.asLong.longCode(cb) * r.asLong.longCode(cb)
                      case (_: SFloat32, _: SFloat32) =>
                        l.asFloat.floatCode(cb) * r.asFloat.floatCode(cb)
                      case (_: SFloat64, _: SFloat64) =>
                        l.asDouble.doubleCode(cb) * r.asDouble.doubleCode(cb)
                    }
                  }

                  cb.assign(kLen, lShape(lPType.nDims - 1))
                  cb.assign(element, numericElementType.zero)
                  cb.forLoop(cb.assign(k, 0L), k < kLen, cb.assign(k, k + 1L), {
                    val lElem = leftPVal.loadElement(lIndices, cb)
                    val rElem = rightPVal.loadElement(rIndices, cb)
                    cb.assign(element, numericElementType.add(multiply(lElem.asPCode, rElem.asPCode), element))
                  })

                  PCode(outputPType.elementType, element)
                }
              }
              emitter.emit(cb, outputPType, region)
            }
          }
        }
      case NDArrayInv(nd) =>
        // Based on https://github.com/numpy/numpy/blob/v1.19.0/numpy/linalg/linalg.py#L477-L547
        emitNDArrayColumnMajorStrides(nd).map(cb) { case pNDCode: PNDArrayCode =>
          val pndVal = pNDCode.memoize(cb, "ndarray_inverse_nd")
          val ndPT = pndVal.pt.asInstanceOf[PCanonicalNDArray]

          val shapeArray = pndVal.shapes(cb)
          val stridesArray = ndPT.makeColumnMajorStrides(shapeArray, region, cb)


          assert(shapeArray.length == 2)

          val M = shapeArray(0)
          val N = shapeArray(1)
          val LDA = M

          val dataFirstAddress = pndVal.firstDataAddress(cb)

          val IPIVptype = PCanonicalArray(PInt32Required, true)
          val IPIVaddr = mb.genFieldThisRef[Long]()
          val WORKaddr = mb.genFieldThisRef[Long]()
          val An = mb.newLocal[Int]()

          val INFOdgetrf = mb.newLocal[Int]()
          val INFOdgetri = mb.newLocal[Int]()
          val INFOerror = (fun: String, info: LocalRef[Int]) => (info cne 0)
            .orEmpty(Code._fatal[Unit](const(s"LAPACK error ${fun}. Error code = ").concat(info.toS)))

          cb.append((N cne M).orEmpty(Code._fatal[Unit](const("Can only invert square matrix"))))

          cb.assign(An, (M * N).toI)

          cb.assign(IPIVaddr, IPIVptype.allocate(region, N.toI))
          cb.append(IPIVptype.stagedInitialize(IPIVaddr, N.toI))

          val (aAadrFirstElement, finish) = ndPT.constructDataFunction(shapeArray, stridesArray, cb, region)
          cb.append(Region.copyFrom(dataFirstAddress,
            aAadrFirstElement, An.toL * 8L))

          cb.assign(INFOdgetrf, Code.invokeScalaObject5[Int, Int, Long, Int, Long, Int](LAPACK.getClass, "dgetrf",
            M.toI,
            N.toI,
            aAadrFirstElement,
            LDA.toI,
            IPIVptype.firstElementOffset(IPIVaddr, N.toI)
          ))
          cb.append(INFOerror("dgetrf", INFOdgetrf))

          cb.assign(WORKaddr, Code.invokeStatic1[Memory, Long, Long]("malloc", An.toL * 8L))

          cb.assign(INFOdgetri, Code.invokeScalaObject6[Int, Long, Int, Long, Long, Int, Int](LAPACK.getClass, "dgetri",
            N.toI,
            aAadrFirstElement,
            LDA.toI,
            IPIVptype.firstElementOffset(IPIVaddr, N.toI),
            WORKaddr,
            N.toI
          ))
          cb.append(INFOerror("dgetri", INFOdgetri))

          finish(cb)
        }
      case x@NDArraySVD(nd, full_matrices, computeUV) =>
        emitNDArrayColumnMajorStrides(nd).flatMap(cb){ case ndPCode: PNDArrayCode =>
          val ndPVal = ndPCode.memoize(cb, "nd_svd_value")

          val infoDGESDDResult = cb.newLocal[Int]("infoDGESDD")
          val infoDGESDDErrorTest = (extraErrorMsg: String) => (infoDGESDDResult cne  0)
            .orEmpty(Code._fatal[Unit](const(s"LAPACK error DGESDD. $extraErrorMsg Error code = ").concat(infoDGESDDResult.toS)))

          val LWORKAddress = mb.newLocal[Long]("svd_lwork_address")
          val shapes = ndPVal.shapes(cb)
          val M = shapes(0)
          val N = shapes(1)
          val K = cb.newLocal[Long]("nd_svd_K")
          cb.assign(K, (M < N).mux(M, N))
          val LDA = M
          val LDU = M
          val UCOL: Value[Long] = if (full_matrices) M else K
          val LDVT = if (full_matrices) N else K
          val IWORK = cb.newLocal[Long]("dgesdd_IWORK_address")
          val A = cb.newLocal[Long]("dgesdd_A_address")
          val firstElementDataAddress = ndPVal.firstDataAddress(cb)

          cb.assign(LWORKAddress, Code.invokeStatic1[Memory, Long, Long]("malloc",  8L))

          val (jobz, sPType, uData, uFinisher, vtData, vtFinisher) = if (computeUV) {
            val outputPType = x.pType.asInstanceOf[PTuple]
            val uPType = outputPType.fields(0).typ.asInstanceOf[PCanonicalNDArray]
            val sPType = outputPType.fields(1).typ.asInstanceOf[PCanonicalNDArray]
            val vtPType = outputPType.fields(2).typ.asInstanceOf[PCanonicalNDArray]

            val uShapeSeq = FastIndexedSeq[Value[Long]](M, UCOL)
            val (uData, uFinisher) = uPType.constructDataFunction(uShapeSeq, uPType.makeColumnMajorStrides(uShapeSeq, region, cb), cb, region)
            val vtShapeSeq = FastIndexedSeq[Value[Long]](LDVT, N)
            val (vtData, vtFinisher) = vtPType.constructDataFunction(vtShapeSeq, vtPType.makeColumnMajorStrides(vtShapeSeq, region, cb), cb, region)

            (if (full_matrices) "A" else "S", sPType, uData, uFinisher, vtData, vtFinisher)
          }
          else {
            def noOp(cb: EmitCodeBuilder): SNDArrayCode = { throw new IllegalStateException("Can't happen")}
            ("N", x.pType.asInstanceOf[PCanonicalNDArray], const(0L), noOp(_), const(0L), noOp(_))
          }

          val (sDataAddress, sFinisher) = sPType.constructDataFunction(IndexedSeq(K), sPType.makeColumnMajorStrides(IndexedSeq(K), region, cb), cb, region)

          cb.assign(infoDGESDDResult, Code.invokeScalaObject13[String, Int, Int, Long, Int, Long, Long, Int, Long, Int, Long, Int, Long, Int](LAPACK.getClass, "dgesdd",
            jobz,
            M.toI,
            N.toI,
            A,
            LDA.toI,
            sDataAddress,
            uData,
            LDU.toI,
            vtData,
            LDVT.toI,
            LWORKAddress,
            -1,
            IWORK
          ))

          cb.append(infoDGESDDErrorTest("Failed size query."))

          cb.assign(IWORK, Code.invokeStatic1[Memory, Long, Long]("malloc", K.toL * 8L * 4L)) // 8K 4 byte integers.
          cb.assign(A, Code.invokeStatic1[Memory, Long, Long]("malloc", M * N * 8L))
          // Copy data into A because dgesdd destroys the input array:
          cb.append(Region.copyFrom(firstElementDataAddress, A, (M * N) * 8L))

          def LWORK = Region.loadDouble(LWORKAddress).toI
          val WORK = cb.newLocal[Long]("dgesdd_work_address")

          cb.assign(WORK, Code.invokeStatic1[Memory, Long, Long]("malloc", LWORK.toL * 8L))

          cb.assign(infoDGESDDResult, Code.invokeScalaObject13[String, Int, Int, Long, Int, Long, Long, Int, Long, Int, Long, Int, Long, Int](LAPACK.getClass, "dgesdd",
            jobz,
            M.toI,
            N.toI,
            A,
            LDA.toI,
            sDataAddress,
            uData,
            LDU.toI,
            vtData,
            LDVT.toI,
            WORK,
            LWORK,
            IWORK
          ))

          cb.append(Code.invokeStatic1[Memory, Long, Unit]("free", IWORK.load()))
          cb.append(Code.invokeStatic1[Memory, Long, Unit]("free", A.load()))
          cb.append(Code.invokeStatic1[Memory, Long, Unit]("free", WORK.load()))
          cb.append(Code.invokeStatic1[Memory, Long, Unit]("free", LWORKAddress.load()))

          cb.append(infoDGESDDErrorTest("Failed result computation."))

          val s = sFinisher(cb)

          val resultPCode = if (computeUV) {
            val u = uFinisher(cb)
            val vt = vtFinisher(cb)

            val outputPType = x.pType.asInstanceOf[PCanonicalTuple]
            outputPType.constructFromFields(cb, region, FastIndexedSeq(EmitCode.present(cb.emb, u.asPCode), EmitCode.present(cb.emb, s), EmitCode.present(cb.emb, vt.asPCode)), deepCopy = false)
          } else {
            s
          }
          IEmitCode(cb, false, resultPCode)

        }
      case x@NDArrayQR(nd, mode) =>
        // See here to understand different modes: https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.qr.html
        emitNDArrayColumnMajorStrides(nd).map(cb) { case pndCode: PNDArrayCode =>
          val pndValue = pndCode.memoize(cb, "ndarray_qr_nd")
          // This does a lot of byte level copying currently, so only trust
          // the PCanonicalNDArray representation.
          assert(pndValue.pt.isInstanceOf[PCanonicalNDArray])

          val shapeArray = pndValue.shapes(cb)

          val LWORKAddress = cb.newLocal[Long]("dgeqrf_lwork_address")

          val M = shapeArray(0)
          val N = shapeArray(1)
          val K = new Value[Long] {
            def get: Code[Long] = (M < N).mux(M, N)
          }
          val LDA = new Value[Long] {
            override def get: Code[Long] = (M > 1L).mux(M, 1L) // Possible stride tricks could change this in the future.
          }

          def LWORK = (Region.loadDouble(LWORKAddress).toI > 0).mux(Region.loadDouble(LWORKAddress).toI, 1)

          val ndPT = pndValue.pt.asInstanceOf[PCanonicalNDArray]
          val dataFirstElementAddress = pndValue.firstDataAddress(cb)

          val hPType = ndPT
          val hShapeArray = FastIndexedSeq[Value[Long]](N, M)
          val hStridesArray = hPType.makeRowMajorStrides(hShapeArray, region, cb)
          val (hFirstElement, hFinisher) = hPType.constructDataFunction(hShapeArray, hStridesArray, cb, region)

          val tauNDPType = PCanonicalNDArray(PFloat64Required, 1, true)
          val (tauFirstElementAddress, tauFinisher) = tauNDPType.constructDataFunction(IndexedSeq(K), IndexedSeq(const(8L)), cb, region)
          val workAddress = cb.newLocal[Long]("ndarray_qr_workAddress")

          val aNumElements = cb.newLocal[Long]("ndarray_qr_aNumElements")

          val infoDGEQRFResult = cb.newLocal[Int]("ndaray_qr_infoDGEQRFResult")
          val infoDGEQRFErrorTest = (extraErrorMsg: String) => (infoDGEQRFResult cne 0)
            .orEmpty(Code._fatal[Unit](const(s"LAPACK error DGEQRF. $extraErrorMsg Error code = ").concat(infoDGEQRFResult.toS)))

          // Computing H and Tau
          cb.assign(aNumElements, ndPT.numElements(shapeArray))
          cb.append(Region.copyFrom(dataFirstElementAddress, hFirstElement, (M * N) * 8L))

          cb.assign(LWORKAddress, region.allocate(8L, 8L))

          cb.assign(infoDGEQRFResult, Code.invokeScalaObject7[Int, Int, Long, Int, Long, Long, Int, Int](LAPACK.getClass, "dgeqrf",
            M.toI,
            N.toI,
            hFirstElement,
            LDA.toI,
            tauFirstElementAddress,
            LWORKAddress,
            -1
          ))
          cb.append(infoDGEQRFErrorTest("Failed size query."))

          cb.assign(workAddress, Code.invokeStatic1[Memory, Long, Long]("malloc", LWORK.toL * 8L))
          cb.assign(infoDGEQRFResult, Code.invokeScalaObject7[Int, Int, Long, Int, Long, Long, Int, Int](LAPACK.getClass, "dgeqrf",
            M.toI,
            N.toI,
            hFirstElement,
            LDA.toI,
            tauFirstElementAddress,
            workAddress,
            LWORK
          ))
          cb.append(Code.invokeStatic1[Memory, Long, Unit]("free", workAddress.load()))
          cb.append(infoDGEQRFErrorTest("Failed to compute H and Tau."))

          val h = hFinisher(cb)
          val hMemo = h.memoize(cb, "ndarray_qr_h_memo")

          val result: PCode = if (mode == "raw") {
            val resultType = x.pType.asInstanceOf[PCanonicalBaseStruct]
            val rawPType = x.pType.asInstanceOf[PTuple]
            assert(hPType equalModuloRequired rawPType.types(0).asInstanceOf[PCanonicalNDArray],  s"hPType = ${hPType}, other = ${rawPType.types(0).asInstanceOf[PCanonicalNDArray]}")
            val tau = tauFinisher(cb)

            resultType.constructFromFields(cb, region, FastIndexedSeq(
              EmitCode.present(cb.emb, hMemo),
              EmitCode.present(cb.emb, tau)
            ), deepCopy = false)

          } else {
            val (rPType, rRows, rCols) = if (mode == "r") {
              (x.pType.asInstanceOf[PCanonicalNDArray], K, N)
            } else if (mode == "complete") {
              (x.pType.asInstanceOf[PTuple].types(1).asInstanceOf[PCanonicalNDArray], M, N)
            } else if (mode == "reduced") {
              (x.pType.asInstanceOf[PTuple].types(1).asInstanceOf[PCanonicalNDArray], K, N)
            } else {
              throw new AssertionError(s"Unsupported QR mode $mode")
            }

            val rShapeArray = FastIndexedSeq[Value[Long]](rRows, rCols)

            val rStridesArray = rPType.makeColumnMajorStrides(rShapeArray, region, cb)

            val (rDataAddress, rFinisher) = rPType.constructDataFunction(rShapeArray, rStridesArray, cb, region)

            // This block assumes that `rDataAddress` and `aAddressDGEQRF` point to column major arrays.
            // TODO: Abstract this into ndarray ptype/pcode interface methods.
            val currRow = cb.newLocal[Long]("ndarray_qr_currRow")
            val currCol = cb.newLocal[Long]("ndarray_qr_currCol")

            val curWriteAddress = cb.newLocal[Long]("ndarray_qr_curr_write_addr", rDataAddress)

            // I think this just copies out the upper triangle into new ndarray in column major order
            cb.forLoop({cb.assign(currCol, 0L)}, currCol < rCols, {cb.assign(currCol, currCol + 1L)}, {
              cb.forLoop({cb.assign(currRow, 0L)}, currRow < rRows, {cb.assign(currRow, currRow + 1L)}, {
                cb.append(Region.storeDouble(
                  curWriteAddress,
                  (currCol >= currRow).mux(
                    hMemo.loadElement(IndexedSeq(currCol, currRow), cb).asDouble.doubleCode(cb),
                    0.0
                  )
                ))
                cb.assign(curWriteAddress, curWriteAddress + rPType.elementType.byteSize)
              })
            })

            val computeR = rFinisher(cb)

            if (mode == "r") {
              computeR
            }
            else {
              val crPType = x.pType.asInstanceOf[PCanonicalTuple]

              val qPType = crPType.types(0).asInstanceOf[PCanonicalNDArray]
              val qShapeArray = if (mode == "complete") Array(M, M) else Array(M, K)
              val qStridesArray = qPType.makeColumnMajorStrides(qShapeArray, region, cb)

              val infoDORGQRResult = cb.newLocal[Int]("ndarray_qr_DORGQR_info")
              val infoDORQRErrorTest = (extraErrorMsg: String) => (infoDORGQRResult cne 0)
                .orEmpty(Code._fatal[Unit](const(s"LAPACK error DORGQR. $extraErrorMsg Error code = ").concat(infoDORGQRResult.toS)))

              val qCondition = cb.newLocal[Boolean]("ndarray_qr_qCondition")
              val numColsToUse = cb.newLocal[Long]("ndarray_qr_numColsToUse")
              val aAddressDORGQRFirstElement = cb.newLocal[Long]("ndarray_qr_dorgqr_a")

              val qNumElements = cb.newLocal[Long]("ndarray_qr_qNumElements")


              val rNDArray = computeR.memoize(cb, "ndarray_qr_rNDAArrayAddress")
              cb.assign(qCondition, const(mode == "complete") && (M > N))
              cb.assign(numColsToUse, qCondition.mux(M, K))
              cb.assign(qNumElements, M * numColsToUse)

              cb.ifx(qCondition, {
                cb.assign(aAddressDORGQRFirstElement, region.allocate(8L, qNumElements * ndPT.elementType.byteSize))
                cb.append(Region.copyFrom(hFirstElement,
                  aAddressDORGQRFirstElement, aNumElements * 8L))
              }, {
                // We are intentionally clobbering h, since we aren't going to return it to anyone.
                cb.assign(aAddressDORGQRFirstElement, hFirstElement)
              })

              cb.assign(infoDORGQRResult, Code.invokeScalaObject8[Int, Int, Int, Long, Int, Long, Long, Int, Int](LAPACK.getClass, "dorgqr",
                M.toI,
                numColsToUse.toI,
                K.toI,
                aAddressDORGQRFirstElement,
                LDA.toI,
                tauFirstElementAddress,
                LWORKAddress,
                -1
              ))
              cb.append(infoDORQRErrorTest("Failed size query."))
              cb.append(workAddress := Code.invokeStatic1[Memory, Long, Long]("malloc", LWORK.toL * 8L))
              cb.assign(infoDORGQRResult, Code.invokeScalaObject8[Int, Int, Int, Long, Int, Long, Long, Int, Int](LAPACK.getClass, "dorgqr",
                M.toI,
                numColsToUse.toI,
                K.toI,
                aAddressDORGQRFirstElement,
                LDA.toI,
                tauFirstElementAddress,
                workAddress,
                LWORK
              ))
              cb.append(Code.invokeStatic1[Memory, Long, Unit]("free", workAddress.load()))
              cb.append(infoDORQRErrorTest("Failed to compute Q."))
              val (qFirstElementAddress, qFinisher) = qPType.constructDataFunction(qShapeArray, qStridesArray, cb, region)
              cb.append(Region.copyFrom(aAddressDORGQRFirstElement,
                qFirstElementAddress, (M * numColsToUse) * 8L))

              crPType.constructFromFields(cb, region, FastIndexedSeq(
                EmitCode.present(cb.emb, qFinisher(cb)),
                EmitCode.present(cb.emb, rNDArray)
              ), deepCopy = false)
            }
          }
          result
        }
      case x: NDArrayMap  =>  emitDeforestedNDArrayI(x)
      case x: NDArrayMap2 =>  emitDeforestedNDArrayI(x)
      case x: NDArrayReshape => emitDeforestedNDArrayI(x)
      case x: NDArrayConcat => emitDeforestedNDArrayI(x)
      case x: NDArraySlice => emitDeforestedNDArrayI(x)
      case x: NDArrayFilter => emitDeforestedNDArrayI(x)
      case x: NDArrayAgg => emitDeforestedNDArrayI(x)
      case x@RunAgg(body, result, states) =>
        val newContainer = AggContainer.fromBuilder(cb, states.toArray, "run_agg")
        emitVoid(body, container = Some(newContainer))
        val codeRes = emitI(result, container = Some(newContainer))

        codeRes.map(cb) { pc =>
          val res = cb.memoizeField(pc, "agg_res")
          newContainer.cleanup()
          res
        }

      case x@ResultOp(start, sig) =>
        val AggContainer(aggs, sc, _) = container.get

        val pt = x.pType.asInstanceOf[PCanonicalTuple]

        val addr = cb.newLocal("resultop_tuple_addr", pt.allocate(region))
        cb += pt.stagedInitialize(addr, setMissing = false)

        (0 until aggs.length).foreach { j =>
          val idx = start + j
          val rvAgg = agg.Extract.getAgg(sig(j))
          val fieldAddr = cb.newLocal(s"resultop_field_addr_$j", pt.fieldOffset(addr, j))
          rvAgg.storeResult(cb, sc.states(idx), pt.types(j), fieldAddr, region,
            (cb: EmitCodeBuilder) => cb += pt.setFieldMissing(addr, j))
        }

        presentPC(pt.loadCheapPCode(cb, addr))
      case x@ApplySeeded(fn, args, seed, rt) =>
        val codeArgs = args.map(a => (a.pType, EmitCode.fromI(cb.emb)(emitInNewBuilder(_, a))))
        val impl = x.implementation
        val unified = impl.unify(Array.empty[Type], args.map(_.typ), rt)
        assert(unified)
        impl.applySeededI(seed, cb, region, pt, codeArgs: _*)

      case AggStateValue(i, _) =>
        val AggContainer(_, sc, _) = container.get
        presentC(sc.states(i).serializeToRegion(cb, coerce[PBinary](pt), region))

      case ToArray(a) =>
        EmitStream.produce(this, a, cb, region, env, container)
          .map(cb) { case stream: SStreamCode => StreamUtils.toArray(cb, stream.producer, region) }

      case x@StreamFold(a, zero, accumName, valueName, body) =>
        EmitStream.produce(this, a, cb, region, env, container)
          .flatMap(cb) { case (stream: SStreamCode) =>
            val producer = stream.producer

            val xAcc = mb.newEmitField(accumName, x.accPType) // in future, will choose compatible type for zero/body with requiredness
            val xElt = mb.newEmitField(valueName, producer.element.pt)

            var tmpRegion: Settable[Region] = null

            if (producer.requiresMemoryManagementPerElement) {
              cb.assign(producer.elementRegion, Region.stagedCreate(Region.REGULAR, region.getPool()))

              tmpRegion = mb.genFieldThisRef[Region]("streamfold_tmpregion")
              cb.assign(tmpRegion, Region.stagedCreate(Region.REGULAR, region.getPool()))

              cb.assign(xAcc, emitI(zero, tmpRegion)
                .map(cb)(pc => pc.castTo(cb, tmpRegion, x.accPType)))
            } else {
              cb.assign(producer.elementRegion, region)
              cb.assign(xAcc, emitI(zero, producer.elementRegion)
                .map(cb)(pc => pc.castTo(cb, producer.elementRegion, x.accPType)))
            }

            producer.unmanagedConsume(cb) { cb =>
              cb.assign(xElt, producer.element)

              if (producer.requiresMemoryManagementPerElement) {
                cb.assign(xAcc, emitI(body, producer.elementRegion, env.bind(accumName -> xAcc, valueName -> xElt))
                  .map(cb)(pc => pc.castTo(cb, tmpRegion, x.accPType, deepCopy = true)))
                cb += producer.elementRegion.clearRegion()
                val swapRegion = cb.newLocal[Region]("streamfold_swap_region", producer.elementRegion)
                cb.assign(producer.elementRegion, tmpRegion.load())
                cb.assign(tmpRegion, swapRegion.load())
              } else {
                cb.assign(xAcc, emitI(body, producer.elementRegion, env.bind(accumName -> xAcc, valueName -> xElt))
                  .map(cb)(pc => pc.castTo(cb, producer.elementRegion, x.accPType, deepCopy = false)))
              }
            }

            if (producer.requiresMemoryManagementPerElement) {
              cb.assign(xAcc, xAcc.toI(cb).map(cb)(pc => pc.castTo(cb, region, pc.pt, deepCopy = true)))
              cb += producer.elementRegion.invalidate()
              cb += tmpRegion.invalidate()
            }
            xAcc.toI(cb)
          }

      case x@StreamFold2(a, acc, valueName, seq, res) =>
        emitStream(a, cb, region)
          .flatMap(cb) { case stream: SStreamCode =>
            val producer = stream.producer

            var tmpRegion: Settable[Region] = null

            val xElt = mb.newEmitField(valueName, producer.element.pt)
            val names = acc.map(_._1)
            val accTypes = x.accPTypes
            val accVars = (names, accTypes).zipped.map(mb.newEmitField)

            val resEnv = env.bind(names.zip(accVars): _*)
            val seqEnv = resEnv.bind(valueName, xElt)

            if (producer.requiresMemoryManagementPerElement) {
              cb.assign(producer.elementRegion, Region.stagedCreate(Region.REGULAR, region.getPool()))

              tmpRegion = mb.genFieldThisRef[Region]("streamfold_tmpregion")
              cb.assign(tmpRegion, Region.stagedCreate(Region.REGULAR, region.getPool()))

              (accVars, acc).zipped.foreach { case (xAcc, (_, x)) =>
                cb.assign(xAcc, emitI(x, tmpRegion).map(cb)(_.castTo(cb, tmpRegion, xAcc.pt)))
              }
            } else {
              cb.assign(producer.elementRegion, region)
              (accVars, acc).zipped.foreach { case (xAcc, (_, x)) =>
                cb.assign(xAcc, emitI(x, region).map(cb)(_.castTo(cb, region, xAcc.pt)))
              }
            }

            producer.unmanagedConsume(cb) { cb =>
              cb.assign(xElt, producer.element)
              if (producer.requiresMemoryManagementPerElement) {
                (accVars, seq).zipped.foreach { (accVar, ir) =>
                  cb.assign(accVar,
                    emitI(ir, producer.elementRegion, env = seqEnv)
                      .map(cb)(pc => pc.castTo(cb, tmpRegion, accVar.pt, deepCopy = true)))
                }
                cb += producer.elementRegion.clearRegion()
                val swapRegion = cb.newLocal[Region]("streamfold2_swap_region", producer.elementRegion)
                cb.assign(producer.elementRegion, tmpRegion.load())
                cb.assign(tmpRegion, swapRegion.load())
              } else {
                (accVars, seq).zipped.foreach { (accVar, ir) =>
                  cb.assign(accVar,
                    emitI(ir, producer.elementRegion, env = seqEnv)
                      .map(cb)(pc => pc.castTo(cb, producer.elementRegion, accVar.pt, deepCopy = false)))
                }
              }
            }

            if (producer.requiresMemoryManagementPerElement) {
              accVars.foreach { xAcc =>
                cb.assign(xAcc, xAcc.toI(cb).map(cb)(pc => pc.castTo(cb, region, pc.pt, deepCopy = true)))
              }
              cb += producer.elementRegion.invalidate()
              cb += tmpRegion.invalidate()
            }
            emitI(res, env = resEnv)
          }

      case x@ShuffleWith(
        keyFields,
        rowType,
        rowEType,
        keyEType,
        name,
        writerIR,
        readersIR
      ) =>
        val shuffleType = x.shuffleType
        val shufflePType = x.shufflePType

        val shuffle = CodeShuffleClient.createValue(cb, mb.ecb.getType(shuffleType))

        cb.append(shuffle.start())

        val uuid = SCanonicalShufflePointerSettable.fromArrayBytes(
          cb, region, shufflePType, shuffle.uuid())

        val shuffleEnv = env.bind(name -> mb.newPresentEmitSettable(uuid.pt, uuid))

        val successfulShuffleIds: PValue = emitI(writerIR, env = shuffleEnv)
          .get(cb, "shuffle ID must be non-missing")
          // just store it so the writer gets run
          .memoize(cb, "shuffleSuccessfulShuffleIds")

        val shuffleReaders =
          emitI(readersIR, env = shuffleEnv).memoize(cb, "shuffleReaders")

        cb.append(shuffle.stop())
        cb.append(shuffle.close())

        shuffleReaders.toI(cb)

      case ShuffleWrite(idIR, rowsIR) =>
        val rows = emitStream(rowsIR, cb, region)
          .get(cb, "rows stream was missing in shuffle write")
          .asStream
        val producer = rows.producer

        val rowPType = producer.element.st.canonicalPType()

        val shuffleType = coerce[TShuffle](idIR.typ)
        val uuid = emitI(idIR)
          .get(cb, "shuffle ID must be non-missing")
          .asInstanceOf[SCanonicalShufflePointerCode]
          .memoize(cb, "shuffleClientUUID")
        val shuffle = CodeShuffleClient.createValue(
          cb,
          mb.ecb.getType(shuffleType),
          uuid.loadBytes(),
          mb.ecb.getPType(rowPType),
          Code._null)
        cb += shuffle.startPut()

        producer.memoryManagedConsume(region, cb) { cb =>
          producer.element.toI(cb).consume(cb,
            cb._fatal(s"empty row in shuffle put"),
            sc => cb += shuffle.putValue(rowPType.store(cb, producer.elementRegion, sc, deepCopy = false)))
        }

        cb += shuffle.putValueDone()
        cb += shuffle.endPut()
        cb += shuffle.close()

        val resPType = pt.asInstanceOf[PCanonicalBinary]
        // FIXME: server needs to send uuid for the successful partition
        val boff = cb.memoize(resPType.loadCheapPCode(cb, resPType.allocate(region, 0)), "shuffleWriteBOff")
        val baddr = SingleCodePCode.fromPCode(cb, boff, region)
        cb += resPType.storeLength(baddr.code.asInstanceOf[Code[Long]], 0)
        presentPC(boff)

      case x@ReadValue(path, spec, requestedType) =>
        emitI(path).map(cb) { pv =>
          val ib = cb.newLocal[InputBuffer]("read_ib")
          cb.assign(ib, spec.buildCodeInputBuffer(mb.open(pv.asString.loadString(), checkCodec = true)))
          spec.encodedType.buildDecoder(requestedType, mb.ecb)(cb, region, ib)
        }

      case WriteValue(value, path, spec) =>
        emitI(path).flatMap(cb) { case p: PStringCode =>
          val pv = p.memoize(cb, "write_path")
          emitI(value).map(cb) { v =>
            val ob = cb.newLocal[OutputBuffer]("write_ob")
            cb.assign(ob, spec.buildCodeOutputBuffer(mb.create(pv.asString.loadString())))
            spec.encodedType.buildEncoder(v.st, cb.emb.ecb)
              .apply(cb, v.memoize(cb, "write_value"), ob)
            cb += ob.invoke[Unit]("close")
            pv
          }
        }

      case x@TailLoop(name, args, body) =>
        val loopStartLabel = CodeLabel()
        val inits = args.zip(x.accPTypes)

        val stagedPool = cb.newLocal[RegionPool]("tail_loop_pool_ref")
        cb.assign(stagedPool, region.getPool())
        val loopRef = LoopRef(cb, loopStartLabel, inits.map { case ((name, _), pt) => (name, pt) }, stagedPool)

        val argEnv = env
          .bind((args.map(_._1), loopRef.loopArgs).zipped.toArray: _*)

        val newLoopEnv = loopEnv.getOrElse(Env.empty)

        // Emit into LoopRef's current region. (region 1)
        loopRef.loopArgs.zip(inits).foreach { case (settable, ((_, x), pt)) =>
          settable.store(cb, emitI(x, loopRef.r1).map(cb)(_.castTo(cb, loopRef.r1, pt)))
        }

        cb.define(loopStartLabel)

        emitI(body, env = argEnv, loopEnv = Some(newLoopEnv.bind(name, loopRef))).map(cb) { pc =>
          val answerInRightRegion = pc.copyToRegion(cb, region)
          cb.append(loopRef.r1.clearRegion())
          cb.append(loopRef.r2.clearRegion())
          answerInRightRegion
        }

      case Recur(name, args, _) =>
        val loopRef = loopEnv.get.lookup(name)

        // Need to emit into region 2, clear region 1, then swap them.
        (loopRef.tmpLoopArgs, loopRef.loopTypes, args).zipped.map { case (tmpLoopArg, pt, arg) =>
          tmpLoopArg.store(cb, emitI(arg, loopEnv = None, region = loopRef.r2).map(cb)(_.castTo(cb, loopRef.r2, pt)))
        }

        cb.append(loopRef.r1.clearRegion())

        // Swap
        val temp = cb.newLocal[Region]("recur_temp_swap_region")
        cb.assign(temp, loopRef.r1)
        cb.assign(loopRef.r1, loopRef.r2)
        cb.assign(loopRef.r2, temp)

        cb.assign(loopRef.loopArgs, loopRef.tmpLoopArgs.load())
        cb.goto(loopRef.L)

        // Dead code. The dead label is necessary because you can't append anything else to a code builder
        // after a goto.
        val deadLabel = CodeLabel()
        cb.define(deadLabel)
        IEmitCode.missing(cb, pt.defaultValue(cb.emb))

      case x@CollectDistributedArray(contexts, globals, cname, gname, body, tsd) =>
        val ctxsType = coerce[PStream](contexts.pType)
        val parentCB = mb.ecb

        val functionID: String = {
          val bodyFB = EmitFunctionBuilder[Region, Array[Byte], Array[Byte], Array[Byte]](ctx.executeContext, "collect_distributed_array")

          // FIXME this is terrible
          val m = MakeTuple.ordered(FastSeq(body))
          val bodyReturnPType = PCanonicalTuple(true, body.pType)
          m._pType = bodyReturnPType

          bodyFB.emitWithBuilder { cb =>
            val ctxIB = cb.newLocal[InputBuffer]("cda_ctx_ib", x.contextSpec.buildCodeInputBuffer(
              Code.newInstance[ByteArrayInputStream, Array[Byte]](bodyFB.getCodeParam[Array[Byte]](2))))
            val gIB = cb.newLocal[InputBuffer]("cda_g_ib", x.globalSpec.buildCodeInputBuffer(
              Code.newInstance[ByteArrayInputStream, Array[Byte]](bodyFB.getCodeParam[Array[Byte]](3))))

            val decodedContext = x.contextSpec.encodedType.buildDecoder(x.contextSpec.encodedVirtualType, bodyFB.ecb)
              .apply(cb, bodyFB.getCodeParam[Region](1), ctxIB)
              .asBaseStruct
              .memoize(cb, "decoded_context_tuple")
              .loadField(cb, 0)
              .typecast[PCode]
              .memoize(cb, "decoded_context")

            val decodedGlobal = x.globalSpec.encodedType.buildDecoder(x.globalSpec.encodedVirtualType, bodyFB.ecb)
              .apply(cb, bodyFB.getCodeParam[Region](1), gIB)
              .asBaseStruct
              .memoize(cb, "decoded_global_tuple")
              .loadField(cb, 0)
              .typecast[PCode]
              .memoize(cb, "decoded_global")

            val env = Env[EmitValue](
              (cname, decodedContext),
              (gname, decodedGlobal))

            val bodyResult = new Emit(ctx, bodyFB.ecb)
              .emitI(m, cb, env, None)
              .get(cb, "cda return cannot be missing!")
              .memoize(cb, "cda_body_result")

            val bOS = cb.newLocal[ByteArrayOutputStream]("cda_baos", Code.newInstance[ByteArrayOutputStream]())
            val bOB = cb.newLocal[OutputBuffer]("cda_ob", x.bodySpec.buildCodeOutputBuffer(bOS))
            x.bodySpec.encodedType.buildEncoder(bodyResult.st, cb.emb.ecb)
                .apply(cb, bodyResult, bOB)
            cb += bOB.invoke[Unit]("flush")
            cb += bOB.invoke[Unit]("close")
            bOS.invoke[Array[Byte]]("toByteArray")
          }

          val fID = genUID()
          parentCB.addModule(fID, bodyFB.resultWithIndex())
          fID
        }

        val spark = parentCB.backend()

        val baos = mb.genFieldThisRef[ByteArrayOutputStream]()
        val buf = mb.genFieldThisRef[OutputBuffer]()
        val ctxab = mb.genFieldThisRef[ByteArrayArrayBuilder]()
        val encRes = mb.genFieldThisRef[Array[Array[Byte]]]()

        def wrapInTuple(cb: EmitCodeBuilder, et: EmitCode): SBaseStructPointerCode = {
          PCanonicalTuple(true, et.pt).constructFromFields(cb, region, FastIndexedSeq(et), deepCopy = false)
        }

        def addContexts(cb: EmitCodeBuilder, ctxStream: StreamProducer): Unit = {
          ctxStream.memoryManagedConsume(region, cb, setup = { cb =>
            cb += ctxab.invoke[Int, Unit]("ensureCapacity", ctxStream.length.getOrElse(16))
          }) { cb =>
            cb += baos.invoke[Unit]("reset")
            val ctxTuple = wrapInTuple(cb, ctxStream.element)
              .memoize(cb, "cda_add_contexts_addr")
            x.contextSpec.encodedType.buildEncoder(ctxTuple.st, parentCB)
              .apply(cb, ctxTuple, buf)
            cb += buf.invoke[Unit]("flush")
            cb += ctxab.invoke[Array[Byte], Unit]("add", baos.invoke[Array[Byte]]("toByteArray"))
          }
        }

        def addGlobals(cb: EmitCodeBuilder): Unit = {
          val g = wrapInTuple(cb, EmitCode.fromI(mb)(cb => emitInNewBuilder(cb, globals))).memoize(cb, "cda_g")
          x.globalSpec.encodedType.buildEncoder(g.st, parentCB)
              .apply(cb, g, buf)
          cb += buf.invoke[Unit]("flush")
        }

        def decodeResult(cb: EmitCodeBuilder): PCode = {
          val len = mb.newLocal[Int]("cda_result_length")
          val ib = mb.newLocal[InputBuffer]("decode_ib")

          cb.assign(len, encRes.length())
          x.pType.asInstanceOf[PCanonicalArray].constructFromElements(cb, region, len, deepCopy = false) { (cb, i) =>
            cb.assign(ib, x.bodySpec.buildCodeInputBuffer(Code.newInstance[ByteArrayInputStream, Array[Byte]](encRes(i))))
            val eltTupled = x.bodySpec.encodedType.buildDecoder(x.bodySpec.encodedVirtualType, parentCB)
              .apply(cb, region, ib)
              .asBaseStruct
              .memoize(cb, "cda_eltTupled")
            eltTupled.loadField(cb, 0)
          }
        }

        emitStream(contexts, cb, region).map(cb) { case ctxStream: SStreamCode =>
          cb.assign(baos, Code.newInstance[ByteArrayOutputStream]())
          cb.assign(buf, x.contextSpec.buildCodeOutputBuffer(baos)) // TODO: take a closer look at whether we need two codec buffers?
          cb.assign(ctxab, Code.newInstance[ByteArrayArrayBuilder, Int](16))
          addContexts(cb, ctxStream.producer)
          cb += baos.invoke[Unit]("reset")
          addGlobals(cb)
          cb.assign(encRes, spark.invoke[BackendContext, String, Array[Array[Byte]], Array[Byte], Option[TableStageDependency], Array[Array[Byte]]](
            "collectDArray",
            mb.getObject(ctx.executeContext.backendContext),
            functionID,
            ctxab.invoke[Array[Array[Byte]]]("result"),
            baos.invoke[Array[Byte]]("toByteArray"),
            mb.getObject(tsd)))
          decodeResult(cb)
        }

      case _ =>
        emitFallback(ir)
    }

    if (result.pt != pt) {
      if (!result.pt.equalModuloRequired(pt))
        throw new RuntimeException(s"ptype mismatch:\n  emitted:  ${ result.pt }\n  inferred: ${ ir.pType }\n  ir: $ir")
      (result.pt.required, pt.required) match {
        case (true, false) => result.map(cb)(pc => PCode(pc.pt.setRequired(pt.required), pc.code))
        case (false, true) => IEmitCode.present(cb, result.get(cb))
      }
    } else result
  }

  /**
    * Invariants of the Returned Triplet
    * ----------------------------------
    *
    * The elements of the triplet are called (precompute, missingness, value)
    *
    *  1. evaluate each returned Code[_] at most once
    *  2. evaluate precompute *on all static code-paths* leading to missingness or value
    *  3. guard the the evaluation of value by missingness
    *
    * Triplets returning values cannot have side-effects.  For void triplets, precompute
    * contains the side effect, missingness is false, and value is {@code Code._empty}.
    *
    * JVM gotcha:
    * a variable must be initialized on all static code-paths prior to its use (ergo defaultValue)
    *
    * Argument Convention
    * -------------------
    *
    * {@code In(i)} occupies two argument slots, one for the value and one for a
    * missing bit. The value for {@code  In(0)} is passed as argument
    * {@code  nSpecialArguments + 1}. The missingness bit is the subsequent
    * argument. In general, the value for {@code  In(i)} appears at
    * {@code  nSpecialArguments + 1 + 2 * i}.
    *
    * There must always be at least one special argument: a {@code  Region} in
    * which the IR can allocate memory.
    *
    * When compiling an aggregation expression, {@code AggIn} refers to the first
    * argument {@code In(0)} whose type must be of type
    * {@code tAggIn.elementType}.  {@code tAggIn.symTab} is not used by Emit.
    *
    **/
  private[ir] def emit(ir: IR, mb: EmitMethodBuilder[C], env: E, container: Option[AggContainer]): EmitCode = {
    val region = mb.getCodeParam[Region](1)
    emit(ir, mb, region, env, container, None)
  }

  private[ir] def emitWithRegion(ir: IR, mb: EmitMethodBuilder[C], region: Value[Region], env: E, container: Option[AggContainer]): EmitCode =
    emit(ir, mb, region, env, container, None)

  private def emit(
    ir: IR,
    mb: EmitMethodBuilder[C],
    region: Value[Region],
    env: E,
    container: Option[AggContainer],
    loopEnv: Option[Env[LoopRef]],
    fallingBackFromEmitI: Boolean = false
  ): EmitCode = {

    def emit(ir: IR, region: Value[Region] = region, env: E = env, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): EmitCode =
      this.emit(ir, mb, region, env, container, loopEnv)

    def emitInMethod(ir: IR, mb: EmitMethodBuilder[C]): EmitCode =
      this.emit(ir, mb, Env.empty, container)

    def emitI(ir: IR, cb: EmitCodeBuilder, env: E = env, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): IEmitCode =
      this.emitI(ir, cb, region, env, container, loopEnv)

    def emitVoid(ir: IR, env: E = env, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): Code[Unit] = {
      EmitCodeBuilder.scopedVoid(mb) { cb =>
        this.emitVoid(cb, ir, region, env, container, loopEnv)
      }
    }

    def emitStream(ir: IR, outerRegion: Value[Region] ): EmitCode =
      EmitCode.fromI(mb)(cb => EmitStream.produce(this, ir, cb, outerRegion, env, container))

    val pt = ir.pType

    // ideally, emit would not be called with void values, but initOp args can be void
    // working towards removing this
    if (pt == PVoid)
      return EmitCode.fromI(mb) { cb =>
        this.emitVoid(cb, ir, region, env, container, loopEnv)
        IEmitCode.present(cb, PCode._empty)
      }

    val result: EmitCode = (ir: @unchecked) match {

      case Let(name, value, body) =>
        EmitCode.fromI(mb) { cb =>
          val xVal = if (value.typ.isInstanceOf[TStream]) emitStream(value, region) else emit(value)
          cb.withScopedMaybeStreamValue(xVal, s"let_$name") { ev =>
            emitI(body, cb, env = env.bind(name, ev))
          }
        }

      case Ref(name, _) =>
        val ev = env.lookup(name)
        if (!ev.pt.equalModuloRequired(pt))
          throw new RuntimeException(s"PValue type did not match inferred ptype:\n name: $name\n  pv: ${ ev.pt }\n  ir: $pt")
        ev.load

      case x@(_: ArraySort | _: ToSet | _: ToDict) =>
        val resultTypeAsIterable = coerce[PIterable](x.pType)
        val eltType = x.children(0).asInstanceOf[IR].pType.asInstanceOf[PIterable].elementType
        val eltVType = eltType.virtualType

        val vab = new StagedArrayBuilder(resultTypeAsIterable.elementType, mb, 0)
        val sorter = new ArraySorter(EmitRegion(mb, region), vab)

        val (array, lessThan, distinct, leftRightComparatorNames: Array[String]) = (x: @unchecked) match {
          case ArraySort(a, l, r, lessThan) => (a, lessThan, Code._empty, Array(l, r))
          case ToSet(a) =>
            val discardNext = mb.genEmitMethod("discardNext",
              FastIndexedSeq[ParamType](typeInfo[Region], PCodeEmitParamType(eltType), PCodeEmitParamType(eltType)),
              typeInfo[Boolean])
            val cmp2 = ApplyComparisonOp(EQWithNA(eltVType), In(0, PCodeEmitParamType(eltType)), In(1, PCodeEmitParamType(eltType)))
            InferPType(cmp2)
            val EmitCode(s, m, pv) = emitInMethod(cmp2, discardNext)
            discardNext.emitWithBuilder { cb =>
              cb += s
              m || pv.asBoolean.boolCode(cb)
            }
            val lessThan = ApplyComparisonOp(Compare(eltVType), In(0, PCodeEmitParamType(eltType)), In(1, PCodeEmitParamType(eltType))) < 0
            InferPType(lessThan)
            (a, lessThan, sorter.distinctFromSorted { (r, v1, m1, v2, m2) =>
              EmitCodeBuilder.scopedCode[Boolean](mb) { cb =>
                cb.invokeCode[Boolean](discardNext, r,
                  EmitCode(Code._empty, m1, PCode(eltType, v1)),
                  EmitCode(Code._empty, m2, PCode(eltType, v2)))
              }
            }, Array.empty[String])
          case ToDict(a) =>
            val (k0, k1, keyType) = eltType match {
              case t: PStruct => (GetField(In(0, PCodeEmitParamType(eltType)), "key"), GetField(In(1, PCodeEmitParamType(eltType)), "key"), t.fieldType("key"))
              case t: PTuple => (GetTupleElement(In(0, PCodeEmitParamType(eltType)), 0), GetTupleElement(In(1, PCodeEmitParamType(eltType)), 0), t.types(0))
            }
            val discardNext = mb.genEmitMethod("discardNext",
              FastIndexedSeq[ParamType](typeInfo[Region], PCodeEmitParamType(eltType), PCodeEmitParamType(eltType)),
              typeInfo[Boolean])

            val cmp2 = ApplyComparisonOp(EQWithNA(keyType.virtualType), k0, k1).deepCopy()
            InferPType(cmp2)
            val EmitCode(s, m, pv) = emitInMethod(cmp2, discardNext)
            discardNext.emitWithBuilder { cb =>
              cb += s
              m || pv.asBoolean.boolCode(cb)
            }
            val lessThan = (ApplyComparisonOp(Compare(keyType.virtualType), k0, k1) < 0).deepCopy()
            InferPType(lessThan)
            (a, lessThan, Code(sorter.pruneMissing, sorter.distinctFromSorted { (r, v1, m1, v2, m2) =>
              EmitCodeBuilder.scopedCode[Boolean](mb) { cb =>
                cb.invokeCode[Boolean](discardNext, r,
                  EmitCode(Code._empty, m1, PCode(eltType, v1)),
                  EmitCode(Code._empty, m2, PCode(eltType, v2)))
              }
            }), Array.empty[String])
        }

        val sort = vab.ti match {
          case BooleanInfo => sorter.sort(makeDependentSortingFunction[Boolean](
            region, eltType, lessThan, env, leftRightComparatorNames))
          case IntInfo => sorter.sort(makeDependentSortingFunction[Int](region, eltType, lessThan, env, leftRightComparatorNames))
          case LongInfo => sorter.sort(makeDependentSortingFunction[Long](
            region, eltType, lessThan, env, leftRightComparatorNames))
          case FloatInfo => sorter.sort(makeDependentSortingFunction[Float](
            region, eltType, lessThan, env, leftRightComparatorNames))
          case DoubleInfo => sorter.sort(makeDependentSortingFunction[Double](
            region, eltType, lessThan, env, leftRightComparatorNames))
        }

        val optStream = emitStream(array, region)
        EmitCode.fromI(mb)(cb => optStream.toI(cb).map(cb) { case stream: SStreamCode =>
          StreamUtils.writeToArrayBuilder(cb, stream.producer, vab, region)
          cb += sort
          cb += distinct
          sorter.toRegion(cb, x.pType)
        })

      case CastToArray(a) =>
        val et = emit(a)
        EmitCode(et.setup, et.m, PCode(pt, et.v))

      case In(i, expectedPType) =>
        // this, Code[Region], ...
        val ev = mb.getEmitParam(2 + i, region)
        ev
      case Die(m, typ, errorId) =>
        val cm = emit(m)
        EmitCode(
          Code(
            cm.setup,
            Code._throw[HailException, Unit](Code.newInstance[HailException, String, Int](
              cm.m.mux[String](
                "<exception message missing>",
                coerce[String](StringFunctions.wrapArg(EmitRegion(mb, region), m.pType)(cm.v))), errorId))),
          true,
          pt.defaultValue(mb))

      case ir@Apply(fn, typeArgs, args, rt) =>
        val impl = ir.implementation
        val unified = impl.unify(typeArgs, args.map(_.typ), rt)
        assert(unified)

        val argPTypes = args.map(_.pType)
        val k = (fn, typeArgs, argPTypes, pt)
        val meth =
          methods.get(k) match {
            case Some(funcMB) =>
              funcMB
            case None =>
              val funcMB = impl.getAsMethod(mb.ecb, pt, typeArgs, argPTypes: _*)
              methods.update(k, funcMB)
              funcMB
          }
        val vars = args.map { a => coerce[Any](mb.newLocal()(typeToTypeInfo(a.pType))) }
        EmitCode.fromI(mb) { cb =>
          val emitArgs = args.map(a => EmitCode.fromI(cb.emb)(emitI(a, _))).toFastIndexedSeq
          IEmitCode.multiMapEmitCodes(cb, emitArgs) { codeArgs =>
            for ((l, i) <- vars.zip(codeArgs)) {
              cb.assign(l, i.code)
            }
            PCode(pt, meth.invokeCode[Any](CodeParam(region) +: vars.map(_.get: Param): _*))
          }
        }
      case x@ApplySpecial(_, typeArgs, args, rt) =>
        val codeArgs = args.map(a => emit(a))
        val impl = x.implementation
        val unified = impl.unify(typeArgs, args.map(_.typ), rt)
        assert(unified)
        impl.apply(EmitRegion(mb, region), pt, typeArgs, codeArgs: _*)

      case x@WritePartition(stream, pctx, writer) =>
        val ctxCode = emit(pctx)
        val streamCode = emitStream(stream, region)
        EmitCode.fromI(mb) { cb =>
          streamCode.toI(cb).flatMap(cb) { case stream: SStreamCode =>
            writer.consumeStream(ctx.executeContext, cb, stream.producer, ctxCode, region)
          }
        }

      case x =>
        if (fallingBackFromEmitI) {
          fatal(s"ir is not defined in emit or emitI $x")
        }
        EmitCode.fromI(mb) { cb =>
          emitI(ir, cb)
        }
    }

    if (result.pt != pt) {
      if (!result.pt.equalModuloRequired(pt))
        throw new RuntimeException(s"ptype mismatch:\n  emitted:  ${ result.pt }\n  inferred: ${ ir.pType }\n  ir: $ir")
      (result.pt.required, pt.required) match {
        case (true, false) => EmitCode.fromI(mb)(cb => result.toI(cb).map(cb)(pc => PCode(pc.pt.setRequired(pt.required), pc.code)))
        case (false, true) => EmitCode.fromI(mb) { cb => IEmitCode.present(cb, result.toI(cb).get(cb)) }
      }
    } else result
  }

  private def capturedReferences(ir: IR): (IR, (Emit.E, DependentEmitFunctionBuilder[_]) => Emit.E) = {
    var ids = Set[String]()

    VisitIR(ir) {
      case Ref(id, _) =>
        ids += id
      case _ =>
    }

    (ir, { (env: Emit.E, f: DependentEmitFunctionBuilder[_]) =>
      Env[EmitValue](ids.toFastSeq.flatMap { id =>
         env.lookupOption(id).map { e =>
           (id, f.newDepEmitField(e.load))
        }
      }: _*)
    })
  }

  private def makeDependentSortingFunction[T: TypeInfo](
    region: Code[Region],
    elemPType: PType, ir: IR, env: Emit.E, leftRightComparatorNames: Array[String]): DependentEmitFunctionBuilder[AsmFunction2[T, T, Boolean]] = {
    val (newIR, getEnv) = capturedReferences(ir)
    val f = cb.genDependentFunction[T, T, Boolean](baseName = "sort_compare")
    val fregion = f.newDepField[Region](region)
    var newEnv = getEnv(env, f)

    val leftEC = EmitCode(Code._empty, false, PCode(elemPType, f.getCodeParam[T](1)))
    val rightEC = EmitCode(Code._empty, false, PCode(elemPType, f.getCodeParam[T](2)))
    val sort = f.genEmitMethod("sort",
      FastIndexedSeq(typeInfo[Region], leftEC.emitParamType, rightEC.emitParamType),
      BooleanInfo)

    if (leftRightComparatorNames.nonEmpty) {
      assert(leftRightComparatorNames.length == 2)
      newEnv = newEnv.bindIterable(
        IndexedSeq(
          (leftRightComparatorNames(0), sort.getEmitParam(2, fregion)),
          (leftRightComparatorNames(1), sort.getEmitParam(3, fregion))))
    }

    val EmitCode(setup, m, v) = new Emit(ctx, f.ecb).emit(newIR, sort, newEnv, None)

    sort.emit(Code(setup, m.mux(Code._fatal[Boolean]("Result of sorting function cannot be missing."), v.code)))
    f.apply_method.emitWithBuilder(cb => cb.invokeCode[Boolean](sort, fregion, leftEC, rightEC))
    f
  }

  private def present(pv: PCode): EmitCode = EmitCode(Code._empty, false, pv)

  private def present(pt: PType, c: Code[_]): EmitCode =
    EmitCode(Code._empty, false, PCode(pt, c))

  def deforestNDArrayI(x0: IR, cb: EmitCodeBuilder, region: Value[Region], env: E): IEmitCode = {

    def emit(ir: IR, env: E = env): IEmitCode =
      this.emitI(ir, cb, region, env, None, None)

    def dEmit(ir: IR, env: E = env): IEmitCode = emit(ir, env)

    def deforest(x: IR): IEmitCodeGen[NDArrayEmitter] = {
      val xType = coerce[PNDArray](x.pType)
      val outputNDims = xType.nDims

      x match {
        case NDArrayMap(child, elemName, body) =>
          deforest(child).map(cb) { childEmitter =>
            val childP = child.pType.asInstanceOf[PNDArray]
            val elemPType = childP.elementType

            new NDArrayEmitter(childEmitter.outputShape) {
              override def outputElement(cb: EmitCodeBuilder, idxVars: IndexedSeq[Value[Long]]): PCode = {
                val elemRef = cb.emb.newPresentEmitField("ndarray_map_element_name", elemPType)

                cb.assign(elemRef, childEmitter.outputElement(cb, idxVars))
                val bodyEnv = env.bind(elemName, elemRef)
                val bodyI = dEmit(body, bodyEnv)

                bodyI.get(cb, "NDArray map body cannot be missing")
              }
            }
          }
        case NDArrayMap2(lChild, rChild, lName, rName, body) =>
          deforest(lChild).flatMap(cb) { leftChildEmitter =>
            deforest(rChild).map(cb) { rightChildEmitter =>
              val lP = coerce[PNDArray](lChild.pType)
              val rP = coerce[PNDArray](rChild.pType)

              val leftShapeValues = leftChildEmitter.outputShape
              val rightShapeValues = rightChildEmitter.outputShape

              val (newSetupShape, shapeArray) = NDArrayEmitter.unifyShapes2(cb.emb, leftShapeValues, rightShapeValues)

              cb.append(newSetupShape)

              new NDArrayEmitter(shapeArray) {
                override def outputElement(cb: EmitCodeBuilder, idxVars: IndexedSeq[Value[Long]]): PCode = {
                  val lElemRef = cb.emb.newPresentEmitField(lName, lP.elementType)
                  val rElemRef = cb.emb.newPresentEmitField(rName, rP.elementType)

                  val bodyEnv = env.bind(lName, lElemRef)
                    .bind(rName, rElemRef)

                  val lIdxVars2 = NDArrayEmitter.zeroBroadcastedDims2(cb.emb, idxVars, nDims, leftShapeValues)
                  val rIdxVars2 = NDArrayEmitter.zeroBroadcastedDims2(cb.emb, idxVars, nDims, rightShapeValues)

                  cb.assign(lElemRef, leftChildEmitter.outputElement(cb, lIdxVars2))
                  cb.assign(rElemRef, rightChildEmitter.outputElement(cb, rIdxVars2))

                  dEmit(body, bodyEnv).get(cb, "NDArrayMap2 body cannot be missing")
                }
              }
            }
          }
        case NDArrayReindex(child, indexExpr) =>
          deforest(child).map(cb) { childEmitter =>
            val childPType = child.pType.asInstanceOf[PNDArray]

            val shapeSeq = indexExpr.map { childIndex =>
              if (childIndex < childPType.nDims)
                childEmitter.outputShape(childIndex)
              else
                const(1L)
            }

            new NDArrayEmitter(shapeSeq) {
              override def outputElement(cb: EmitCodeBuilder, idxVars: IndexedSeq[Value[Long]]): PCode = {
                val concreteIdxsForChild = Array.tabulate(childEmitter.nDims) { childDim =>
                  val parentDim = indexExpr.indexOf(childDim)
                  idxVars(parentDim)
                }
                childEmitter.outputElement(cb, concreteIdxsForChild)
              }
            }
          }
        case x@NDArrayReshape(childND, shape) =>
          deforest(childND).flatMap(cb) { childEmitter =>
            val outputNDims = x.pType.nDims

            val childShapeValues = childEmitter.outputShape

            val requestedShapeValues = Array.tabulate(x.pType.nDims)(i => cb.newLocal[Long](s"ndarray_reindex_request_shape_$i")).toIndexedSeq

            dEmit(shape, env).map(cb) { pc =>
              val tupleCode = pc.asBaseStruct
              val tupleValue = tupleCode.memoize(cb, "ndarray_reshape_requested")

              val hasNegativeOne = cb.newLocal[Boolean]("ndarray_reshape_has_neg_one")
              val runningProduct = cb.newLocal[Long]("ndarray_reshape_running_product")
              val replacesNegativeOne = cb.newLocal[Long]("ndarray_reshape_replaces_neg_one")
              val tempShapeElement = cb.newLocal[Long]("ndarray_reshape_temp_shape_element")

              cb.assign(hasNegativeOne, false)
              cb.assign(runningProduct, 1L)

              (0 until outputNDims).foreach { i =>
                cb.assign(tempShapeElement, tupleValue.loadField(cb, i).get(cb, "Can't reshape if elements of reshape tuple are missing.").asLong.longCode(cb))
                cb.ifx(tempShapeElement < 0L,
                  {
                    cb.ifx(tempShapeElement ceq -1L,
                      {
                        cb.ifx(hasNegativeOne, {
                          cb._fatal("Can't infer shape, more than one -1")
                        }, {
                          cb.assign(hasNegativeOne, true)
                        })
                      },
                      {
                        cb._fatal("Can't reshape, new shape must contain only nonnegative numbers or -1")
                      }
                    )
                  },
                  {
                    cb.assign(runningProduct, runningProduct * tempShapeElement)
                  }
                )
              }

              val numElements = cb.newLocal[Long]("ndarray_reshape_child_num_elements")
              cb.assign(numElements, childND.pType.asInstanceOf[PNDArray].numElements(childShapeValues))

              cb.ifx(hasNegativeOne.mux(
                (runningProduct ceq 0L) || (numElements % runningProduct) > 0L,
                numElements cne runningProduct
              ), {
                cb._fatal("Can't reshape since requested shape is incompatible with number of elements")
              })
              cb.assign(replacesNegativeOne, (runningProduct ceq 0L).mux(0L, numElements / runningProduct))

              (0 until outputNDims).foreach { i =>
                cb.assign(tempShapeElement, tupleValue.loadField(cb, i).get(cb, "Can't reshape if elements of reshape tuple are missing.").asLong.longCode(cb))
                cb.assign(requestedShapeValues(i), (tempShapeElement ceq -1L).mux(replacesNegativeOne, tempShapeElement))
              }

              new NDArrayEmitter(requestedShapeValues) {
                override def outputElement(cb: EmitCodeBuilder, idxVars: IndexedSeq[Value[Long]]): PCode = {
                  val storeElementIndex = cb.newLocal[Long]("ndarray_reshape_index_store")
                  cb.assign(storeElementIndex, LinalgCodeUtils.linearizeIndicesRowMajor(idxVars, requestedShapeValues, cb.emb))

                  val (newIdxVarsSetup, newIdxVars) = LinalgCodeUtils.unlinearizeIndexRowMajor(storeElementIndex, childShapeValues, cb.emb)
                  cb.append(newIdxVarsSetup)
                  assert(newIdxVars.length == childEmitter.nDims)

                  childEmitter.outputElement(cb, newIdxVars)
                }
              }
            }
          }
        case x@NDArrayFilter(child, filters) =>
          deforest(child).map(cb) { childEmitter =>

            val filterWasMissing = (0 until filters.size).map(i => cb.newField[Boolean](s"ndarray_filter_${i}_was_missing"))
            val filtPValues = new Array[PIndexableValue](filters.size)
            val outputShape = childEmitter.outputShape.map(_ => cb.newField[Long]("ndarray_filter_output_shapes"))

            filters.zipWithIndex.foreach { case (filt, i) =>
              // Each filt is a sequence that may be missing with elements that may not be missing.
              emit(filt).consume(cb,
                {
                  cb.assign(outputShape(i), childEmitter.outputShape(i))
                  cb.assign(filterWasMissing(i), true)
                },
                {
                  filtArrayPC => {
                    val filtArrayPValue = filtArrayPC.asIndexable.memoize(cb, s"ndarray_filt_array_${i}")
                    filtPValues(i) = filtArrayPValue
                    cb.assign(outputShape(i), filtArrayPValue.loadLength().toL)
                    cb.assign(filterWasMissing(i), false)
                  }
                }
              )
            }

            new NDArrayEmitter(outputShape) {
              override def outputElement(cb: EmitCodeBuilder, idxVars: IndexedSeq[Value[Long]]): PCode = {
                val newIdxVars: IndexedSeq[Settable[Long]] = Array.tabulate(x.pType.nDims) { _ => cb.newField[Long]("ndarray_filter_new_idx_val") }
                newIdxVars.zipWithIndex.foreach { case (newIdxVar, i) =>
                  cb.ifx(filterWasMissing(i), {
                    cb.assign(newIdxVar, idxVars(i))
                  },
                    {
                      cb.assign(newIdxVar, filtPValues(i).loadElement(cb, idxVars(i).toI).get(cb, s"NDArrayFilter: can't filter on missing index (axis=$i)").asLong.longCode(cb))
                    })
                }

                childEmitter.outputElement(cb, newIdxVars)
              }
            }
          }
        case NDArraySlice(child, slicesIR) =>
          deforest(child).flatMap(cb) { childEmitter =>
            emit(slicesIR).flatMap(cb) { slicesPC =>
              val slicesValue = slicesPC.asBaseStruct.memoize(cb, "ndarray_slice_tuple_pv")

              val (indexingIndices, slicingIndices) = slicesValue.pt.types.zipWithIndex.partition { case (pFieldType, idx) =>
                pFieldType.isPrimitive
              } match {
                case (a, b) => (a.map(_._2), b.map(_._2))
              }

              IEmitCode.multiFlatMap[Int, SCode, NDArrayEmitter](indexingIndices, indexingIndex => slicesValue.loadField(cb, indexingIndex), cb) { indexingSCodes =>
                val indexingValues = indexingSCodes.map(sCode => sCode.memoize(cb, "ndarray_slice_indexer"))
                val slicingValueTriples = new BoxedArrayBuilder[(Value[Long], Value[Long], Value[Long])]()
                val outputShape = {
                  IEmitCode.multiFlatMap[Int, SCode, IndexedSeq[Value[Long]]](slicingIndices,
                    valueIdx => slicesValue.loadField(cb, valueIdx), cb) { sCodeSlices: IndexedSeq[SCode] =>
                    IEmitCode.multiFlatMap(sCodeSlices, { sCodeSlice: SCode =>
                      val sValueSlice = sCodeSlice.asBaseStruct.memoize(cb, "ndarray_slice_sCodeSlice")
                      // I know I have a tuple of three elements here, start, stop, step

                      val newDimSizeI = sValueSlice.loadField(cb, 0).flatMap(cb) { startC =>
                        sValueSlice.loadField(cb, 1).flatMap(cb) { stopC =>
                          sValueSlice.loadField(cb, 2).map(cb) { stepC =>
                            val start = cb.newLocal[Long]("ndarray_slice_start", startC.asLong.longCode(cb))
                            val stop = cb.newLocal[Long]("ndarray_slice_stop", stopC.asLong.longCode(cb))
                            val step = cb.newLocal[Long]("ndarray_slice_step", stepC.asLong.longCode(cb))

                            slicingValueTriples.push((start, stop, step))

                            val newDimSize = cb.newLocal[Long]("new_dim_size")
                            cb.ifx(step >= 0L && start <= stop, {
                              cb.assign(newDimSize, const(1L) + ((stop - start) - 1L) / step)
                            }, {
                              cb.ifx(step < 0L && start >= stop, {
                                cb.assign(newDimSize, (((stop - start) + 1L) / step) + 1L)
                              }, {
                                cb.assign(newDimSize, 0L)
                              })
                            })

                            newDimSize

                          }
                        }
                      }
                      newDimSizeI
                    }, cb)(x => IEmitCode(cb, false, x))
                  }
                }

                outputShape.map(cb) { outputShapeSeq =>
                  new NDArrayEmitter(outputShapeSeq) {
                    override def outputElement(cb: EmitCodeBuilder, idxVars: IndexedSeq[Value[Long]]): PCode = {
                      // Iterate through the slices tuple given in. For each single integer, should just copy that integer into
                      // an indexed seq. For each range, should use start and step to modify.
                      val oldIdxVarsIterator = idxVars.toIterator
                      val indexingIterator = indexingValues.toIterator
                      val slicingIterator = slicingValueTriples.result().toIterator

                      val newIdxVars = slicesValue.pt.types.map { fieldType => fieldType match {
                        case indexer: PInt64 => {
                          indexingIterator.next().asPValue.value.asInstanceOf[Value[Long]]
                        }
                        case slicer: PBaseStruct => {
                          val (start, stop, step) = slicingIterator.next()

                          cb.memoize(PCode.apply(PInt64Required, start + oldIdxVarsIterator.next() * step), "ndarray_slice_adjusted_lookup").value.asInstanceOf[Value[Long]]
                        }
                      }}

                      childEmitter.outputElement(cb, newIdxVars)
                    }
                  }
                }
              }
            }
          }
        case NDArrayConcat(nds, axis) =>
          emit(nds).flatMap(cb) { ndsPCode =>
            val ndsArrayPValue = ndsPCode.asIndexable.memoize(cb, "ndarray_concat_array_of_nds")
            val arrLength = ndsArrayPValue.loadLength()
            cb.ifx(arrLength ceq 0, {
              cb._fatal("need at least one ndarray to concatenate")
            })

            val missing = cb.newLocal[Boolean]("ndarray_concat_result_missing")
            cb.assign(missing, false)
            // Need to check if the any of the ndarrays are missing.
            val missingCheckLoopIdx = cb.newLocal[Int]("ndarray_concat_missing_check_idx")
            cb.forLoop(cb.assign(missingCheckLoopIdx, 0), missingCheckLoopIdx < arrLength, cb.assign(missingCheckLoopIdx, missingCheckLoopIdx + 1),
              cb.assign(missing, missing | ndsArrayPValue.isElementMissing(missingCheckLoopIdx))
            )

            IEmitCode(cb, missing, {
              val loopIdx = cb.newLocal[Int]("ndarray_concat_shape_check_idx")
              val firstND = ndsArrayPValue.loadElement(cb, 0).map(cb) { sCode => sCode.asNDArray }.get(cb).memoize(cb, "ndarray_concat_input_0")
              val newShape = (0 until outputNDims).map { dimIdx =>
                val localDim = cb.newLocal[Long](s"ndarray_concat_output_shape_element_${dimIdx}")
                val ndShape = firstND.shapes(cb)
                cb.assign(localDim, ndShape(dimIdx))
                cb.forLoop(cb.assign(loopIdx, 1), loopIdx < arrLength, cb.assign(loopIdx, loopIdx + 1), {
                  val shapeOfNDAtIdx = ndsArrayPValue.loadElement(cb, loopIdx).map(cb) { sCode => sCode.asNDArray }.get(cb).shape(cb).memoize(cb, "ndarray_concat_input_shape")
                  val dimLength = shapeOfNDAtIdx.loadField(cb, dimIdx).get(cb).toPCode(cb, region).memoize(cb, "dimLength").value.asInstanceOf[Value[Long]]

                  if (dimIdx == axis) {
                    cb.assign(localDim, localDim + dimLength)
                  }
                  else {
                    cb.ifx(dimLength.cne(localDim),
                      cb._fatal(const(s"NDArrayConcat: mismatched dimensions of input NDArrays along axis ").concat(loopIdx.toS).concat(": expected ")
                        .concat(localDim.toS).concat(", got ")
                        .concat(dimLength.toS))
                    )
                  }
                })
                localDim
              }

              new NDArrayEmitter(newShape) {
                override def outputElement(cb: EmitCodeBuilder, idxVars: IndexedSeq[Value[Long]]): PCode = {
                  val concatAxisIdx = cb.newLocal[Long]("ndarray_concat_axis_id")
                  val whichNDArrayToRead = cb.newLocal[Int]("ndarray_concat_outputElement_i")

                  cb.assign(concatAxisIdx, idxVars(axis))
                  cb.assign(whichNDArrayToRead, 0)
                  val condition = EmitCodeBuilder.scopedCode[Boolean](cb.emb) { cb =>
                    (concatAxisIdx >= ndsArrayPValue.loadElement(cb, whichNDArrayToRead).get(cb).asNDArray.shape(cb).memoize(cb, "ndarray_concat_condition").loadField(cb, axis).get(cb).asLong.longCode(cb))
                  }
                  cb.whileLoop(condition, {
                    cb.assign(concatAxisIdx, concatAxisIdx - ndsArrayPValue.loadElement(cb, whichNDArrayToRead).get(cb).asNDArray.shape(cb).memoize(cb, "ndarray_concat_output_subtract").loadField(cb, axis).get(cb).asLong.longCode(cb))
                    cb.assign(whichNDArrayToRead, whichNDArrayToRead + 1)
                  })
                  cb.ifx(whichNDArrayToRead >= arrLength, cb._fatal(const("NDArrayConcat: trying to access element greater than length of concatenation axis: ").concat(whichNDArrayToRead.toS).concat(" > ").concat((arrLength - 1).toS)))
                  val transformedIdxs = Array.tabulate(nDims) { idx =>
                    if (idx == axis) concatAxisIdx else idxVars(idx)
                  }.toFastIndexedSeq
                  ndsArrayPValue.loadElement(cb, whichNDArrayToRead).get(cb).asNDArray.memoize(cb, "ndarray_to_load_element_from").loadElement(transformedIdxs, cb).toPCode(cb, region)
                }
              }
            })
          }
        case NDArrayAgg(child, axesToSumOut) =>
          deforest(child).map(cb) { childEmitter =>
            val childDims = child.typ.asInstanceOf[TNDArray].nDims
            val axesToKeep = (0 until childDims).filter(axis => !axesToSumOut.contains(axis))
            val newOutputShape = axesToKeep.map(idx => childEmitter.outputShape(idx))
            val newOutputShapeComplement = axesToSumOut.map(idx => childEmitter.outputShape(idx))

            new NDArrayEmitter(newOutputShape) {
              override def outputElement(cb: EmitCodeBuilder, idxVars: IndexedSeq[Value[Long]]): PCode = {
                val numericElementType = coerce[PNumeric](child.pType.asInstanceOf[PNDArray].elementType)
                val runningSum = NumericPrimitives.newLocal(cb, "ndarray_agg_running_sum", numericElementType.virtualType)
                cb.assign(runningSum, numericElementType.zero)

                SNDArray.forEachIndex(cb, newOutputShapeComplement, "NDArrayAgg_Sum_loop"){ case (cb, coordsBeingSummedOut) =>
                  // Build the new list we need to pass down into child
                  val idxVarsIt = idxVars.toIterator
                  val summedOutIt = coordsBeingSummedOut.toIterator
                  val fullIndicesForChild = (0 until childDims).map(idx =>
                    if (axesToSumOut.contains(idx)) summedOutIt.next() else idxVarsIt.next()
                  )
                  cb.assign(runningSum, numericElementType.add(runningSum, childEmitter.outputElement(cb, fullIndicesForChild).code))
                }

                PCode.apply(numericElementType, runningSum)
              }
            }
          }
        case _ =>
          val ndI = emit(x)
          ndI.map(cb){ ndPCode =>
            val ndPv = ndPCode.asNDArray.memoize(cb, "deforestNDArray_fall_through_ndarray")
            val shape = ndPv.shapes(cb)

            new NDArrayEmitter(shape) {
              override def outputElement(cb: EmitCodeBuilder, idxVars: IndexedSeq[Value[Long]]): PCode = {
                ndPv.asInstanceOf[PNDArrayValue].loadElement(idxVars, cb).toPCode(cb, region)
              }
            }
          }
      }
    }

    deforest(x0).map(cb)(emitter => emitter.emit(cb, coerce[PCanonicalNDArray](x0.pType), region))
  }
}

object NDArrayEmitter {

  def zeroBroadcastedDims2(mb: EmitMethodBuilder[_], loopVars: IndexedSeq[Value[Long]], nDims: Int, shapeArray: IndexedSeq[Value[Long]]): IndexedSeq[Value[Long]] = {
    val broadcasted = 0L
    val notBroadcasted = 1L
    Array.tabulate(nDims)(dim => new Value[Long] {
      def get: Code[Long] = (shapeArray(dim) > 1L).mux(notBroadcasted, broadcasted) * loopVars(dim)
    })
  }

  def broadcastMask(shapeArray: IndexedSeq[Code[Long]]): IndexedSeq[Value[Long]] = {
    val broadcasted = 0L
    val notBroadcasted = 1L
    shapeArray.map(shapeElement => new Value[Long] {
      def get: Code[Long] = (shapeElement > 1L).mux(notBroadcasted, broadcasted)
    })
  }

  def zeroBroadcastedDims(indices: IndexedSeq[Code[Long]], broadcastMask: IndexedSeq[Code[Long]]): IndexedSeq[Value[Long]] = {
    indices.zip(broadcastMask).map { case (index, flag) => new Value[Long] {
      def get: Code[Long] = index * flag
    }}
  }

  def unifyShapes2(mb: EmitMethodBuilder[_], leftShape: IndexedSeq[Value[Long]], rightShape: IndexedSeq[Value[Long]]): (Code[Unit], IndexedSeq[Value[Long]]) = {
    val sb = SetupBuilder(mb)

    val shape = leftShape.zip(rightShape).zipWithIndex.map { case ((left, right), i) =>
      val notSameAndNotBroadcastable = !((left ceq right) || (left ceq 1L) || (right ceq 1L))
      sb.memoizeField(
        notSameAndNotBroadcastable.mux(
          Code._fatal[Long](rightShape.foldLeft[Code[String]](
            leftShape.foldLeft[Code[String]](
              const("Incompatible NDArrayshapes: [ ")
            )((accum, v) => accum.concat(v.toS).concat(" "))
              .concat("] vs [ ")
          )((accum, v) => accum.concat(v.toS).concat(" "))
            .concat("]")),
          (left > right).mux(left, right)),
        s"unify_shapes2_shape$i")
    }

    (sb.result(), shape)
  }

  def matmulShape(cb: EmitCodeBuilder, leftShape: IndexedSeq[Value[Long]], rightShape: IndexedSeq[Value[Long]]): IndexedSeq[Value[Long]] = {
    val mb = cb.emb
    val sb = SetupBuilder(mb)

    assert(leftShape.nonEmpty)
    assert(rightShape.nonEmpty)

    var lK: Value[Long] = null
    var rK: Value[Long] = null
    var shape: IndexedSeq[Value[Long]] = null
    var setup: Code[Unit] = Code._empty

    if (leftShape.length == 1) {
      lK = leftShape.head
      if (rightShape.length == 1) {
        rK = rightShape.head
        shape = FastIndexedSeq()
      } else {
        rK = rightShape(rightShape.length - 2)
        shape = rightShape.slice(0, rightShape.length - 2) :+ rightShape.last
      }
    } else {
      lK = leftShape.last
      if (rightShape.length == 1) {
        rK = rightShape.head
        shape = leftShape.slice(0, leftShape.length - 1)
      } else {
        rK = rightShape(rightShape.length - 2)
        val (unifiedSetup, unifiedShape) = unifyShapes2(mb,
          leftShape.slice(0, leftShape.length - 2),
          rightShape.slice(0, rightShape.length - 2))
        setup = Code(setup, unifiedSetup)
        shape = unifiedShape :+ leftShape(leftShape.length - 2) :+ rightShape.last
      }
    }

    val leftShapeString = const("(").concat(leftShape.map(_.toS).reduce((a, b) => a.concat(", ").concat(b))).concat(")")
    val rightShapeString = const("(").concat(rightShape.map(_.toS).reduce((a, b) => a.concat(", ").concat(b))).concat(")")


    setup = Code(setup,
      (lK cne rK).orEmpty(
        Code._fatal[Unit](const("Matrix dimensions incompatible: ")
          .concat(leftShapeString)
          .concat(" can't be multiplied by matrix with dimensions ")
          .concat(rightShapeString))))

    cb.append(setup)
    shape
  }
}

abstract class NDArrayEmitter(val outputShape: IndexedSeq[Value[Long]])
{
  val nDims = outputShape.length

  def outputElement(cb: EmitCodeBuilder, idxVars: IndexedSeq[Value[Long]]): PCode

  def emit(cb: EmitCodeBuilder, targetType: PCanonicalNDArray, region: Value[Region]): PCode = {
    val shapeArray = outputShape

    val idx = cb.newLocal[Int]("ndarrayemitter_emitloops_idx", 0)

    val (firstElementAddress, finish) = targetType.constructDataFunction(
      outputShape,
      targetType.makeColumnMajorStrides(shapeArray, region, cb),
      cb,
      region)

    SNDArray.forEachIndex(cb, shapeArray, "ndarrayemitter_emitloops") { case (cb, idxVars) =>
      val element = IEmitCode.present(cb, outputElement(cb, idxVars)).consume(cb,  {cb._fatal("NDArray elements cannot  be missing")}, { elementPc  =>
        targetType.elementType.storeAtAddress(cb, firstElementAddress + (idx.toL * targetType.elementType.byteSize), region, elementPc,  true)
      })
      cb.assign(idx, idx + 1)
    }

    finish(cb)
  }
}

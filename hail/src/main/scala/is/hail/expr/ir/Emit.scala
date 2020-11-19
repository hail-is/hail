package is.hail.expr.ir

import java.io._

import is.hail.HailContext
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend.BackendContext
import is.hail.expr.ir.Emit.E
import is.hail.expr.ir.EmitStream.SizedStream
import is.hail.expr.ir.agg.{AggStateSig, ArrayAggStateSig, GroupedStateSig}
import is.hail.expr.ir.functions.StringFunctions
import is.hail.expr.ir.lowering.TableStageDependency
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer}
import is.hail.linalg.{BLAS, LAPACK, LinalgCodeUtils}
import is.hail.services.shuffler._
import is.hail.types.physical._
import is.hail.types.physical.stypes.concrete.{SCanonicalShufflePointerCode, SCanonicalShufflePointerSettable}
import is.hail.types.physical.stypes.primitives.{SFloat32, SFloat64, SInt32, SInt64}
import is.hail.types.virtual._
import is.hail.utils._

import scala.collection.mutable
import scala.language.{existentials, postfixOps}

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

  def apply[C](ctx: ExecuteContext, ir: IR, fb: EmitFunctionBuilder[C], aggs: Option[Array[AggStateSig]] = None) {
    TypeCheck(ir)

    val mb = fb.apply_method
    val container = aggs.map { a =>
      val c = fb.addAggStates(a)
      AggContainer(a, c, () => ())
    }
    val emitter = new Emit[C](ctx, fb.ecb)
    val region = StagedRegion(mb.getCodeParam[Region](1))
    if (ir.typ == TVoid) {
      fb.emitWithBuilder { cb =>
        emitter.emitVoid(cb, ir, mb, region, Env.empty, container, None)
        Code._empty
      }
    } else {
      fb.emitWithBuilder { cb =>
        emitter.emitI(ir, cb, region, Env.empty, container, None).handle(cb, {
          cb._throw[RuntimeException](
            Code.newInstance[RuntimeException, String]("cannot return empty"))
        }).code
      }
    }
  }
}

object AggContainer {
  // FIXME remove this when EmitStream also has a codebuilder
  def fromVars(aggs: Array[AggStateSig], mb: EmitMethodBuilder[_], region: Settable[Region], off: Settable[Long]): (AggContainer, Code[Unit], Code[Unit]) = {

    val (setup, aggState) = EmitCodeBuilder.scoped(mb) { cb =>
      val states = agg.StateTuple(aggs.map(a => agg.AggStateSig.getState(a, cb.emb.ecb)))
      val aggState = new agg.TupleAggregatorState(mb.ecb, states, region, off)
      cb += (region := Region.stagedCreate(Region.REGULAR))
      cb += region.load().setNumParents(aggs.length)
      cb += (off := region.load().allocate(aggState.storageType.alignment, aggState.storageType.byteSize))
      states.createStates(cb)
      aggState
    }

    val cleanup = EmitCodeBuilder.scopedVoid(mb) { cb =>
      aggState.store(cb)
      cb += region.load().invalidate()
      cb.assign(region, Code._null)
    }

    (AggContainer(aggs, aggState, () => ()), setup, cleanup)
  }

  def fromMethodBuilder[C](aggs: Array[AggStateSig], mb: EmitMethodBuilder[C], varPrefix: String): (AggContainer, Code[Unit], Code[Unit]) =
    fromVars(aggs, mb, mb.genFieldThisRef[Region](s"${varPrefix}_top_region"), mb.genFieldThisRef[Long](s"${varPrefix}_off"))

  def fromBuilder[C](cb: EmitCodeBuilder, aggs: Array[AggStateSig], varPrefix: String): AggContainer = {
    val off = cb.newField[Long](s"${varPrefix}_off")
    val region = cb.newField[Region](s"${varPrefix}_top_region", Region.stagedCreate(Region.REGULAR))
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

  def get: EmitCode
}

class EmitUnrealizableValue(val pt: PType, private val ec: EmitCode) extends EmitValue {
  assert(!pt.isRealizable)
  private var used: Boolean = false

  def get: EmitCode = {
    assert(!used)
    used = true
    ec
  }
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
    val Lmissing = CodeLabel()
    val Lpresent = CodeLabel()
    cb.ifx(m, { cb.goto(Lmissing) })
    val res: A = value
    cb.goto(Lpresent)
    IEmitCodeGen(Lmissing, Lpresent, res)
  }

  def apply[A](Lmissing: CodeLabel, Lpresent: CodeLabel, value: A): IEmitCodeGen[A] =
    IEmitCodeGen(Lmissing, Lpresent, value)

  def present[A](cb: EmitCodeBuilder, value: => A): IEmitCodeGen[A] = {
    val Lpresent = CodeLabel()
    cb.goto(Lpresent)
    IEmitCodeGen(CodeLabel(), Lpresent, value)
  }

  def sequence[A, B, C](seq: IndexedSeq[A], toIec: A => IEmitCodeGen[B], cb: EmitCodeBuilder)
      (f: IndexedSeq[B] => C): IEmitCodeGen[C] = {
    val Lmissing = CodeLabel()
    val Lpresent = CodeLabel()

    val pcs = seq.map { elem =>
      val iec = toIec(elem)

      cb.define(iec.Lmissing)
      cb.goto(Lmissing)
      cb.define(iec.Lpresent)

      iec.value
    }
    val pc = f(pcs)
    cb.goto(Lpresent)

    IEmitCodeGen(Lmissing, Lpresent, pc)
  }

  def flatten[A, B](seq: IndexedSeq[() => IEmitCodeGen[A]], cb: EmitCodeBuilder)(f: IndexedSeq[A] => B): IEmitCodeGen[B] =
    sequence(seq, { (i: () => IEmitCodeGen[A]) => i() }, cb)(f)
}

object IEmitCodeGen {
  implicit class IEmitCode(val iec: IEmitCodeGen[PCode]) extends AnyVal {
    def pc: PCode = iec.value
    def pt: PType = pc.pt

    def memoize(cb: EmitCodeBuilder, name: String): EmitValue =
      cb.memoize(iec, name)
  }
}

case class IEmitCodeGen[+A](Lmissing: CodeLabel, Lpresent: CodeLabel, value: A) {

  // This method is a very temporary patch until we can properly separate SCode and PCode
  def typecast[T]: IEmitCodeGen[T] = IEmitCodeGen(Lmissing, Lpresent, value.asInstanceOf[T])

  def map[B](cb: EmitCodeBuilder)(f: (A) => B): IEmitCodeGen[B] = {
    val Lpresent2 = CodeLabel()
    cb.define(Lpresent)
    val value2 = f(value)
    cb.goto(Lpresent2)
    IEmitCodeGen(Lmissing, Lpresent2, value2)
  }

  def mapMissing(cb: EmitCodeBuilder)(ifMissing: => Unit): IEmitCodeGen[A] = {
    val Lmissing2 = CodeLabel()
    cb.define(Lmissing)
    ifMissing
    cb.goto(Lmissing2)
    IEmitCodeGen(Lmissing2, Lpresent, value)
  }

  def flatMap[B](cb: EmitCodeBuilder)(f: (A) => IEmitCodeGen[B]): IEmitCodeGen[B] = {
    cb.define(Lpresent)
    val ec2 = f(value)
    cb.define(ec2.Lmissing)
    cb.goto(Lmissing)
    IEmitCodeGen(Lmissing, ec2.Lpresent, ec2.value)
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
  def apply(setup: Code[Unit], ec: EmitCode): EmitCode =
    new EmitCode(Code(setup, ec.setup), ec.m, ec.pv)

  def apply(setup: Code[Unit], ev: EmitValue): EmitCode =
    EmitCode(setup, ev.get)

  def present(pt: PType, v: Code[_]): EmitCode = EmitCode(Code._empty, false, PCode(pt, v))

  def present(pc: PCode): EmitCode = EmitCode(Code._empty, false, pc)

  def missing(pt: PType): EmitCode = EmitCode(Code._empty, true, pt.defaultValue)

  def fromI(mb: EmitMethodBuilder[_])(f: (EmitCodeBuilder) => IEmitCode): EmitCode = {
    val cb = EmitCodeBuilder(mb)
    val iec = f(cb)
    val setup = cb.result()
    val newEC = EmitCode(Code._empty,
      new CCode(setup.start, iec.Lmissing.start, iec.Lpresent.start),
      iec.pc)
    iec.Lmissing.clear()
    iec.Lpresent.clear()
    setup.clear()
    newEC
  }

  def codeTupleTypes(pt: PType): IndexedSeq[TypeInfo[_]] = {
    val ts = pt.codeTupleTypes()
    if (pt.required)
      ts
    else
      ts :+ BooleanInfo
  }

  def fromCodeTuple(pt: PType, ct: IndexedSeq[Code[_]]): EmitCode = {
    if (pt.required)
      new EmitCode(Code._empty, const(false), pt.fromCodeTuple(ct))
    else
      new EmitCode(Code._empty, coerce[Boolean](ct.last), pt.fromCodeTuple(ct.init))
  }
}

case class EmitCode(setup: Code[Unit], m: Code[Boolean], pv: PCode) {
  def pt: PType = pv.pt

  def v: Code[_] = pv.code

  def value[T]: Code[T] = coerce[T](v)

  def map(f: PCode => PCode): EmitCode = EmitCode(setup, m, pv = f(pv))

  def toI(cb: EmitCodeBuilder): IEmitCode = {
    val Lmissing = CodeLabel()
    val Lpresent = CodeLabel()
    cb += setup
    cb.ifx(m, { cb.goto(Lmissing) }, { cb.goto(Lpresent) })
    IEmitCode(Lmissing, Lpresent, pv)
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

  def load(): EmitCode = get
}

abstract class PresentEmitSettable extends EmitValue {
  def store(cb: EmitCodeBuilder, pc: PCode): Unit

  def load(): EmitCode = get
}

class RichIndexedSeqEmitSettable(is: IndexedSeq[EmitSettable]) {
  def load(): IndexedSeq[EmitCode] = is.map(_.load())
}

object LoopRef {
  def apply(mb: EmitMethodBuilder[_], L: CodeLabel, args: IndexedSeq[(String, PType)]): LoopRef = {
    val (loopArgs, tmpLoopArgs) = args.zipWithIndex.map { case ((name, pt), i) =>
      (mb.newEmitField(s"$name$i", pt), mb.newEmitField(s"tmp$name$i", pt))
    }.unzip
    LoopRef(L, args.map(_._2), loopArgs, tmpLoopArgs)
  }
}

case class LoopRef(
  L: CodeLabel,
  loopTypes: IndexedSeq[PType],
  loopArgs: IndexedSeq[EmitSettable],
  tmpLoopArgs: IndexedSeq[EmitSettable])

abstract class EstimableEmitter[C] {
  def emit(mb: EmitMethodBuilder[C]): Code[Unit]

  def estimatedSize: Int
}

class Emit[C](
  val ctx: ExecuteContext,
  val cb: EmitClassBuilder[C]) { emitSelf =>

  val methods: mutable.Map[(String, Seq[Type], Seq[PType], PType), EmitMethodBuilder[C]] = mutable.Map()


  private[ir] def emitVoid(cb: EmitCodeBuilder, ir: IR, mb: EmitMethodBuilder[C], region: StagedRegion, env: E, container: Option[AggContainer], loopEnv: Option[Env[LoopRef]]): Unit = {

    def emit(ir: IR, mb: EmitMethodBuilder[C] = mb, region: StagedRegion = region, env: E = env, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): EmitCode =
      this.emit(ir, mb, region, env, container, loopEnv)

    def emitStream(ir: IR, outerRegion: ParentStagedRegion, mb: EmitMethodBuilder[C] = mb): EmitCode =
      EmitStream.emit(ctx, this, ir, mb, outerRegion, env, container)

    def emitVoid(ir: IR, cb: EmitCodeBuilder = cb, mb: EmitMethodBuilder[C] = mb, region: StagedRegion = region, env: E = env, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): Unit =
      this.emitVoid(cb, ir, mb, region, env, container, loopEnv)

    def emitI(ir: IR, region: StagedRegion = region, env: E = env, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): IEmitCode =
      this.emitI(ir, cb, region, env, container, loopEnv)

    (ir: @unchecked) match {
      case Void() =>
        Code._empty

      case Begin(xs) =>
        xs.foreach(x => emitVoid(x))

      case If(cond, cnsq, altr) =>
        assert(cnsq.typ == TVoid && altr.typ == TVoid)

        emitI(cond).consume(cb, {}, m => cb.ifx(m.tcode[Boolean], emitVoid(cnsq), emitVoid(altr)))

      case Let(name, value, body) => value.pType match {
        case streamType: PCanonicalStream =>
          val outerRegion = region.asParent(streamType.separateRegions, "Let value")
          val valuet = emitStream(value, outerRegion)
          val bodyEnv = env.bind(name -> new EmitUnrealizableValue(streamType, valuet))

          emitVoid(body, env = bodyEnv)

        case valueType =>
          val x = mb.newEmitField(name, valueType)
          cb.assign(x, emit(value))
          emitVoid(body, env = env.bind(name, x))
      }

      case StreamFor(a, valueName, body) =>
        val streamType = coerce[PStream](a.pType)
        val eltType = streamType.elementType

        val outerRegion = region.asParent(streamType.separateRegions, "StreamFor")
        val eltRegion = outerRegion.createChildRegion(mb)
        val streamOpt = emitStream(a, outerRegion)

        def forBody(elt: EmitCode): Code[Unit] = {
          val xElt = mb.newEmitField(valueName, eltType)
          val bodyEnv = env.bind(valueName -> xElt)
          EmitCodeBuilder.scopedVoid(mb) { cb =>
            cb.assign(xElt, elt)
            emitVoid(body, cb, env = bodyEnv)
            cb += eltRegion.clear()
          }
        }

        streamOpt.toI(cb).consume(cb,
          {},
          { s =>
            cb += eltRegion.allocateRegion(Region.REGULAR)
            cb += s.asStream.stream.getStream(eltRegion).forEach(ctx, mb, forBody)
            cb += eltRegion.free()
          })

      case x@InitOp(i, args, sig) =>
        val AggContainer(aggs, sc, _) = container.get
        assert(aggs(i) == sig.state)
        val rvAgg = agg.Extract.getAgg(sig)

        val argVars = args.zip(rvAgg.initOpTypes).map { case (a, t) =>
          emit(a, container = container.flatMap(_.nested(i, init = true)))
            .castTo(mb, region.code, t)
        }.toArray

        cb += sc.newState(i)
        rvAgg.initOp(cb, sc.states(i), argVars)

      case x@SeqOp(i, args, sig) =>
        val AggContainer(aggs, sc, _) = container.get
        assert(sig.state == aggs(i))
        val rvAgg = agg.Extract.getAgg(sig)

        val argVars = args.zip(rvAgg.seqOpTypes).map { case (a, t) =>
          emit(a, container = container.flatMap(_.nested(i, init = false)))
            .castTo(mb, region.code, t)
        }.toArray
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

        val init = Code(Array.range(start, start + ns)
          .map(i => sc.newState(i)))

        cb += init
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
        writer.writeMetadata(emitI(annotations), cb, region.code)

      case CombOpValue(i, value, aggSig) =>
        val AggContainer(_, sc, _) = container.get
        val rvAgg = agg.Extract.getAgg(aggSig)
        val tempState = AggStateSig.getState(aggSig.state, mb.ecb)
        val aggStateOffset = mb.genFieldThisRef[Long](s"combOpValue_${i}_state");

        val v = emitI(value)
        v.consume(cb,
          cb._fatal("cannot combOp a missing value"),
          { serializedValue =>
            cb.assign(aggStateOffset, region.code.allocate(tempState.storageType.alignment, tempState.storageType.byteSize))
            tempState.createState(cb)
            cb += tempState.newState()
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
            cb += sc.newState(i)
            sc.states(i).deserializeFromBytes(cb, serializedValue.pt.asInstanceOf[PBinary], serializedValue.code.asInstanceOf[Code[Long]])
          }
        )
    }
  }

  private def emitI(ir: IR, cb: EmitCodeBuilder, region: StagedRegion, env: E,
    container: Option[AggContainer], loopEnv: Option[Env[LoopRef]]
  ): IEmitCode = {
    val mb: EmitMethodBuilder[C] = cb.emb.asInstanceOf[EmitMethodBuilder[C]]

    def emitI(ir: IR, region: StagedRegion = region, env: E = env, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): IEmitCode =
      this.emitI(ir, cb, region, env, container, loopEnv)

    def emitStream(ir: IR, outerRegion: ParentStagedRegion): IEmitCode =
      EmitStream.emit(ctx, this, ir, mb, outerRegion, env, container).toI(cb)

    def emitVoid(ir: IR, env: E = env, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): Unit =
      this.emitVoid(cb, ir: IR, mb, region, env, container, loopEnv)

    def emitFallback(ir: IR, env: E = env, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): IEmitCode =
      this.emit(ir, mb, region, env, container, loopEnv, fallingBackFromEmitI = true).toI(cb)

    def emitDeforestedNDArray(ir: IR): IEmitCode =
      deforestNDArray(ir, mb, region, env).toI(cb)

    def emitNDArrayColumnMajorStrides(ir: IR): IEmitCode = {
      emitI(ir).map(cb){case pNDCode: PNDArrayCode =>
        val pNDValue = pNDCode.memoize(cb, "ndarray_column_major_check")
        val isColumnMajor = LinalgCodeUtils.checkColumnMajor(pNDValue, cb)
        val pAnswer = cb.emb.newPField("ndarray_output_column_major", pNDValue.pt)
        cb.ifx(isColumnMajor, {cb.assign(pAnswer, pNDValue)},
        {
          cb.assign(pAnswer, LinalgCodeUtils.createColumnMajorCode(pNDValue, cb, region.code))
        })
        pAnswer.get
      }
    }

    val pt = ir.pType

    if (pt == PVoid) {
      emitVoid(ir)
      return IEmitCode(CodeLabel(), CodeLabel(), PCode._empty)
    }

    def presentPC(pc: PCode): IEmitCode = {
      val Lpresent = CodeLabel()
      cb.goto(Lpresent)
      IEmitCode(CodeLabel(), Lpresent, pc)
    }

    def presentC(c: Code[_]): IEmitCode = presentPC(PCode(pt, c))

    (ir: @unchecked) match {
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
          allocateAndStoreString(mb, region.code, Code.invokeScalaObject0[String](
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
        IEmitCode(cb, const(true), pt.defaultValue)
      case IsNA(v) =>
        val m = emitI(v).consumeCode(cb, true, _ => false)
        presentC(m)
      case ApplyBinaryPrimOp(op, l, r) =>
        emitI(l).flatMap(cb) { pcL =>
          emitI(r).map(cb)(pcR => PCode(pt, BinaryOp.emit(op, l.typ, r.typ, pcL.code, pcR.code)))
        }
      case ApplyUnaryPrimOp(op, x) =>
        emitI(x).map(cb)(pc => PCode(pt, UnaryOp.emit(op, x.typ, pc.code)))
      case ApplyComparisonOp(op, l, r) =>
        val f = op.codeOrdering(mb, l.pType, r.pType)
        if (op.strict) {
          emitI(l).flatMap(cb)(l => emitI(r).map(cb)(r => PCode(pt, f((false, l.code), (false, r.code)))))
        } else {
          val lc = emitI(l).memoize(cb, "l")
          val rc = emitI(r).memoize(cb, "r")
          presentC(
            f((lc.m, lc.v),
              (rc.m, rc.v)))
        }

      case x@ArrayRef(a, i, s) =>
        val errorTransformer: Code[String] => Code[String] = s match {
          case Str("") =>
            val prettied = Pretty.short(x)
            (c: Code[String]) =>
              c.concat("\n----------\nIR:\n").concat(prettied)
          case Str(s) => (c: Code[String]) => c.concat("\n----------\nPython traceback:\n").concat(s)
          case s =>
            (_c: Code[String]) => {
              val ies = emitI(s)
              val c = cb.newLocal("array_ref_c", _c)
              ies.consume(cb, {}, { pc =>
                cb.assign(c, c.concat("\n----------\nPython traceback:\n")
                        .concat(pc.asString.loadString()))
              })
              c.load()
            }
        }

        emitI(a).flatMap(cb) { (ac) =>
          emitI(i).flatMap(cb) { (ic) =>
            val av = ac.asIndexable.memoize(cb, "aref_a")
            val iv = cb.newLocal("i", ic.asInt.intCode(cb))

            cb.ifx(iv < 0 || iv >= av.loadLength(), {
              cb._fatal(errorTransformer(
                const("array index out of bounds: index=")
                  .concat(iv.toS)
                  .concat(", length=")
                  .concat(av.loadLength().toS)))
            })
            av.loadElement(cb, iv).typecast[PCode]
          }
        }

      case ArrayLen(a) =>
        emitI(a).map(cb) { (ac) =>
          PCode(PInt32Required, ac.asIndexable.loadLength())
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

      case x@MakeNDArray(dataIR, shapeIR, rowMajorIR) =>
        val xP = x.pType
        val dataContainer = dataIR.pType
        val shapePType = coerce[PTuple](shapeIR.pType)
        val dataPType = xP.data.pType
        val nDims = shapePType.size

        emitI(rowMajorIR).flatMap(cb) { isRowMajorCode =>
          emitI(shapeIR).flatMap(cb) { case shapeTupleCode: PBaseStructCode =>
            emitI(dataIR).map(cb) { case dataCode: PIndexableCode =>
              val shapeTupleValue = shapeTupleCode.memoize(cb, "make_ndarray_shape")
              val requiredData = dataPType.store(cb, region.code, dataCode, false)

              (0 until nDims).foreach { index =>
                cb.ifx(shapeTupleValue.isFieldMissing(index),
                  cb.append(Code._fatal[Unit](s"shape missing at index $index")))
              }

              def shapeBuilder(srvb: StagedRegionValueBuilder): Code[Unit] =
                EmitCodeBuilder.scopedVoid(mb) { cb =>
                  cb += srvb.start()
                  (0 until nDims).foreach { index =>
                    val shape =
                      shapeTupleValue.loadField(cb, index).get(cb).asPCode.tcode[Long]
                    cb += srvb.addLong(shape)
                    cb += srvb.advance()
                  }
                }

              def makeStridesBuilder(sourceShape: PBaseStructValue, isRowMajor: Code[Boolean], mb: EmitMethodBuilder[_]): StagedRegionValueBuilder => Code[Unit] = { srvb => EmitCodeBuilder.scopedVoid(mb) { cb =>
                def shapeCodeSeq1 = (0 until nDims).map { i =>
                  sourceShape.loadField(cb, i).get(cb).memoize(cb, s"make_ndarray_shape_${i}").asPValue.asInstanceOf[Value[Long]]
                }

                cb.ifx(isRowMajor, {
                  cb += xP.makeRowMajorStridesBuilder(shapeCodeSeq1, mb)(srvb)
                }, {
                  cb += xP.makeColumnMajorStridesBuilder(shapeCodeSeq1, mb)(srvb)
                })
              }}

              xP.construct(shapeBuilder, makeStridesBuilder(shapeTupleValue, isRowMajorCode.asBoolean.boolCode(cb), mb), requiredData, mb, region.code)
            }
          }
        }
      case NDArrayShape(ndIR) =>
        emitI(ndIR).map(cb){ case pc: PNDArrayCode => pc.shape.asPCode}
      case x@NDArrayReindex(child, indexMap) =>
        val childEC = emitI(child)
        val childPType = coerce[PNDArray](child.pType)
        childEC.map(cb){ case pndCode: PNDArrayCode =>
          val pndVal = pndCode.memoize(cb, "ndarray_reindex_child")
          val childShape = pndVal.shapes(cb)
          val childStrides = pndVal.strides(cb)

          x.pType.construct({ srvb =>
            Code(
              srvb.start(),
              Code.foreach(indexMap) { childIndex =>
                Code(
                  srvb.addLong(if (childIndex < childPType.nDims) childShape(childIndex) else 1L),
                  srvb.advance())
              })
          }, { srvb =>
            Code(
              srvb.start(),
              Code.foreach(indexMap) { index =>
                Code(
                  srvb.addLong(if (index < childPType.nDims) childStrides(index) else 0L),
                  srvb.advance())
              })
          }, childPType.data.load(pndVal.tcode[Long]), mb, region.code)
        }

      case NDArrayRef(nd, idxs, errorId) =>
        val ndt = emitI(nd)
        val ndPType = coerce[PNDArray](nd.pType)

        ndt.flatMap(cb) { case ndCode: PNDArrayCode =>
          IEmitCode.sequence(idxs, (ir: IR) => emitI(ir), cb) { idxPCodes: IndexedSeq[PCode] =>
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
          emitNDArrayColumnMajorStrides(rChild).flatMap(cb) { case rightPCode: PNDArrayCode =>
            val lPType = leftPCode.pt
            val rPType = rightPCode.pt

            val leftPVal = leftPCode.memoize(cb, "left_ndarray_matmul")
            val rightPVal = rightPCode.memoize(cb, "right_ndarray_matmul")

            val lShape = leftPVal.shapes(cb)
            val rShape = rightPVal.shapes(cb)

            val unifiedShape = NDArrayEmitter.matmulShape(cb, lShape, rShape)

            val leftBroadcastMask = if (lPType.nDims > 2) NDArrayEmitter.broadcastMask(lShape) else IndexedSeq[Value[Long]]()
            val rightBroadcastMask = if (rPType.nDims > 2) NDArrayEmitter.broadcastMask(rShape) else IndexedSeq[Value[Long]]()

            val outputPType = PCanonicalNDArray(lPType.elementType, TNDArray.matMulNDims(lPType.nDims, rPType.nDims), true)

            if ((lPType.elementType.isInstanceOf[PFloat64] || lPType.elementType.isInstanceOf[PFloat32]) && lPType.nDims == 2 && rPType.nDims == 2) {
              val leftDataAddress = lPType.data.load(leftPVal.tcode[Long])
              val rightDataAddress = rPType.data.load(rightPVal.tcode[Long])

              val answerPArrayAddress = mb.genFieldThisRef[Long]()
              val M = lShape(lPType.nDims - 2)
              val N = rShape(rPType.nDims - 1)
              val K = lShape(lPType.nDims - 1)

              val LDA = M
              val LDB = K
              val LDC = M

              cb.ifx((M cne 0L) && (N cne 0L) && (K cne 0L), {
                cb.assign(answerPArrayAddress, outputPType.data.pType.allocate(region.code, (M * N).toI))
                cb.append(outputPType.data.pType.stagedInitialize(answerPArrayAddress, (M * N).toI))
                cb.append(lPType.elementType match {
                  case PFloat32(_) =>
                    Code.invokeScalaObject13[String, String, Int, Int, Int, Float, Long, Int, Long, Int, Float, Long, Int, Unit](BLAS.getClass, method = "sgemm",
                      "N",
                      "N",
                      M.toI,
                      N.toI,
                      K.toI,
                      1.0f,
                      lPType.data.pType.firstElementOffset(leftDataAddress),
                      LDA.toI,
                      rPType.data.pType.firstElementOffset(rightDataAddress),
                      LDB.toI,
                      0.0f,
                      outputPType.data.pType.firstElementOffset(answerPArrayAddress, (M * N).toI),
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
                      lPType.data.pType.firstElementOffset(leftDataAddress),
                      LDA.toI,
                      rPType.data.pType.firstElementOffset(rightDataAddress),
                      LDB.toI,
                      0.0,
                      outputPType.data.pType.firstElementOffset(answerPArrayAddress, (M * N).toI),
                      LDC.toI
                    )
                })
              },
                {
                  cb.assign(answerPArrayAddress, outputPType.data.pType.zeroes(mb, region.code, (M * N).toI))
                }
              )
              val res = outputPType.construct(
                outputPType.makeShapeBuilder(IndexedSeq(M, N)),
                outputPType.makeColumnMajorStridesBuilder(IndexedSeq(M, N), mb),
                answerPArrayAddress,
                mb,
                region.code)

              IEmitCode.present(cb, res)
            } else {
              val numericElementType = coerce[PNumeric](lPType.elementType)
              val eVti = typeToTypeInfo(numericElementType)
              val iUnifiedShape = IEmitCodeGen(CodeLabel(), CodeLabel(), unifiedShape)

              val emitter = new NDArrayEmitter2(iUnifiedShape) {
                override def outputElement(cb: EmitCodeBuilder, idxVars: IndexedSeq[Value[Long]]): Code[_] = {
                  val elemMB = cb.emb
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

                  val lElem = leftPVal.loadElement(lIndices, cb)
                  val rElem = rightPVal.loadElement(rIndices, cb)
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

                  Code(
                    k := 0L,
                    kLen := lShape(lPType.nDims - 1),
                    element := numericElementType.zero,
                    Code.whileLoop(k < kLen,
                      element := numericElementType.add(multiply(lElem.asPCode, rElem.asPCode), element),
                      k := k + 1L),
                    element
                  )
                }
              }
              emitter.emit(cb, outputPType, region.code)
            }
          }
        }
      case NDArrayInv(nd) =>
        // Based on https://github.com/numpy/numpy/blob/v1.19.0/numpy/linalg/linalg.py#L477-L547
        emitNDArrayColumnMajorStrides(nd).map(cb) { case pNDCode: PNDArrayCode =>
          val pndVal = pNDCode.memoize(cb, "ndarray_inverse_nd")
          val shapeArray = pndVal.shapes(cb)

          assert(shapeArray.length == 2)

          val M = shapeArray(0)
          val N = shapeArray(1)
          val LDA = M

          val dataAddress = pndVal.pt.data.load(pndVal.tcode[Long])

          val IPIVptype = PCanonicalArray(PInt32Required, true)
          val IPIVaddr = mb.genFieldThisRef[Long]()
          val WORKaddr = mb.genFieldThisRef[Long]()
          val Aaddr = mb.genFieldThisRef[Long]()
          val An = mb.newLocal[Int]()

          val INFOdgetrf = mb.newLocal[Int]()
          val INFOdgetri = mb.newLocal[Int]()
          val INFOerror = (fun: String, info: LocalRef[Int]) => (info cne 0)
            .orEmpty(Code._fatal[Unit](const(s"LAPACK error ${fun}. Error code = ").concat(info.toS)))

          cb.append((N cne M).orEmpty(Code._fatal[Unit](const("Can only invert square matrix"))))

          cb.assign(An, (M * N).toI)
          cb.assign(Aaddr, pndVal.pt.data.pType.allocate(region.code, An))
          cb.append(pndVal.pt.data.pType.stagedInitialize(Aaddr, An))
          cb.append(Region.copyFrom(pndVal.pt.data.pType.firstElementOffset(dataAddress, An),
            pndVal.pt.data.pType.firstElementOffset(Aaddr, An), An.toL * 8L))

          cb.assign(IPIVaddr, IPIVptype.allocate(region.code, N.toI))
          cb.append(IPIVptype.stagedInitialize(IPIVaddr, N.toI))

          cb.assign(INFOdgetrf, Code.invokeScalaObject5[Int, Int, Long, Int, Long, Int](LAPACK.getClass, "dgetrf",
            M.toI,
            N.toI,
            pndVal.pt.data.pType.firstElementOffset(Aaddr, An.toI),
            LDA.toI,
            IPIVptype.firstElementOffset(IPIVaddr, N.toI)
          ))
          cb.append(INFOerror("dgetrf", INFOdgetrf))

          cb.assign(WORKaddr, Code.invokeStatic1[Memory, Long, Long]("malloc", An.toL * 8L))

          cb.assign(INFOdgetri, Code.invokeScalaObject6[Int, Long, Int, Long, Long, Int, Int](LAPACK.getClass, "dgetri",
            N.toI,
            pndVal.pt.data.pType.firstElementOffset(Aaddr, An.toI),
            LDA.toI,
            IPIVptype.firstElementOffset(IPIVaddr, N.toI),
            WORKaddr,
            N.toI
          ))
          cb.append(INFOerror("dgetri", INFOdgetri))

          val shapeBuilder = pndVal.pt.makeShapeBuilder(shapeArray)
          val stridesBuilder = pndVal.pt.makeColumnMajorStridesBuilder(shapeArray, mb)
          
          pndVal.pt.construct(shapeBuilder, stridesBuilder, Aaddr, mb, region.code)
        }
      case x@NDArraySVD(nd, full_matrices, computeUV) =>
        emitNDArrayColumnMajorStrides(nd).flatMap(cb){ case ndPCode: PNDArrayCode =>
          val ndPVal = ndPCode.memoize(cb, "nd_svd_value")
          val ndPType = ndPCode.pt.asInstanceOf[PCanonicalNDArray]

          val infoDGESDDResult = cb.newLocal[Int]("infoDGESDD")
          val infoDGESDDErrorTest = (extraErrorMsg: String) => (infoDGESDDResult cne  0)
            .orEmpty(Code._fatal[Unit](const(s"LAPACK error DGESDD. $extraErrorMsg Error code = ").concat(infoDGESDDResult.toS)))

          val LWORKAddress = mb.newLocal[Long]()

          val shapes = ndPVal.shapes(cb)
          val M = shapes(0)
          val N = shapes(1)
          val K = cb.newLocal[Long]("nd_svd_K")
          cb.assign(K, (M < N).mux(M, N))
          val LDA = M
          val sData = cb.newField[Long]("nd_svd_S_address")
          val U_data = cb.newField[Long]("nd_svd_U_address")
          val LDU = M
          val UCOL: Value[Long] = if (full_matrices) M else K
          val vtData = cb.newField[Long]("nd_svd_VT_address")
          val LDVT = if (full_matrices) N else K
          val IWORK = cb.newLocal[Long]("dgesdd_IWORK_address")
          val A = cb.newLocal[Long]("dgesdd_A_address")
          val dataAddress = cb.newLocal[Long]("nd_svd_dataArray_address")

          cb.assign(dataAddress, ndPVal.pt.data.load(ndPVal.value.asInstanceOf[Value[Long]]))
          cb.assign(LWORKAddress, Code.invokeStatic1[Memory, Long, Long]("malloc",  8L))

          val (jobz, sPType, optUPType, optVTPType) = if (computeUV) {
            val outputPType = x.pType.asInstanceOf[PTuple]
            val uPType = outputPType.fields(0).typ.asInstanceOf[PNDArray]
            val sPType = outputPType.fields(1).typ.asInstanceOf[PNDArray]
            val vtPType = outputPType.fields(2).typ.asInstanceOf[PNDArray]

            cb.assign(U_data, uPType.data.pType.allocate(region.code, UCOL.toI * LDU.toI))
            cb.append(uPType.data.pType.stagedInitialize(U_data, UCOL.toI * LDU.toI))

            cb.assign(vtData, vtPType.data.pType.allocate(region.code, N.toI * LDVT.toI))
            cb.append(vtPType.data.pType.stagedInitialize(vtData, N.toI * LDVT.toI))
            (if (full_matrices) "A" else "S", sPType, Some(uPType), Some(vtPType))
          }
          else {
            cb.assign(U_data, 0L)
            cb.assign(vtData, 0L)
            ("N", x.pType.asInstanceOf[PNDArray], None, None)
          }

          cb.assign(sData, sPType.data.pType.allocate(region.code, K.toI))
          cb.append(sPType.data.pType.stagedInitialize(sData, K.toI))

          def uStart = optUPType.map(uPType => uPType.data.pType.firstElementOffset(U_data)).getOrElse(U_data.get)
          def vtStart = optVTPType.map(vtPType => vtPType.data.pType.firstElementOffset(vtData)).getOrElse(vtData.get)

          cb.assign(infoDGESDDResult, Code.invokeScalaObject13[String, Int, Int, Long, Int, Long, Long, Int, Long, Int, Long, Int, Long, Int](LAPACK.getClass, "dgesdd",
            jobz,
            M.toI,
            N.toI,
            A,
            LDA.toI,
            sPType.data.pType.firstElementOffset(sData),
            uStart,
            LDU.toI,
            vtStart,
            LDVT.toI,
            LWORKAddress,
            -1,
            IWORK
          ))

          cb.append(infoDGESDDErrorTest("Failed size query."))

          cb.assign(IWORK, Code.invokeStatic1[Memory, Long, Long]("malloc", K.toL * 8L * 4L)) // 8K 4 byte integers.
          cb.assign(A, Code.invokeStatic1[Memory, Long, Long]("malloc", M * N * 8L))
          // Copy data into A because dgesdd destroys the input array:
          cb.append(Region.copyFrom(ndPType.data.pType.firstElementOffset(dataAddress, (M * N).toI), A, (M * N) * 8L))

          def LWORK = Region.loadDouble(LWORKAddress).toI
          val WORK = cb.newLocal[Long]("dgesdd_work_address")

          cb.assign(WORK, Code.invokeStatic1[Memory, Long, Long]("malloc", LWORK.toL * 8L))

          cb.assign(infoDGESDDResult, Code.invokeScalaObject13[String, Int, Int, Long, Int, Long, Long, Int, Long, Int, Long, Int, Long, Int](LAPACK.getClass, "dgesdd",
            jobz,
            M.toI,
            N.toI,
            A,
            LDA.toI,
            sPType.data.pType.firstElementOffset(sData),
            uStart,
            LDU.toI,
            vtStart,
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

          val sShapeSeq = FastIndexedSeq[Value[Long]](K)
          val s = sPType.construct(sPType.makeShapeBuilder(sShapeSeq), sPType.makeColumnMajorStridesBuilder(sShapeSeq, mb), sData, mb, region.code)

          val resultPCode = if (computeUV) {
            val uShapeSeq = FastIndexedSeq[Value[Long]](M, UCOL)
            val uPType = optUPType.get
            val u = uPType.construct(uPType.makeShapeBuilder(uShapeSeq), uPType.makeColumnMajorStridesBuilder(uShapeSeq, mb), U_data, mb, region.code)

            val vtShapeSeq = FastIndexedSeq[Value[Long]](LDVT, N)
            val vtPType = optVTPType.get
            val vt = vtPType.construct(vtPType.makeShapeBuilder(vtShapeSeq), vtPType.makeColumnMajorStridesBuilder(vtShapeSeq, mb), vtData, mb, region.code)

            val resultSRVB = new StagedRegionValueBuilder(mb, x.pType, region.code)
            cb.append(Code(
              resultSRVB.start(),
              resultSRVB.addIRIntermediate(u),
              resultSRVB.advance(),
              resultSRVB.addIRIntermediate(s),
              resultSRVB.advance(),
              resultSRVB.addIRIntermediate(vt),
              resultSRVB.advance())
            )

            PCode.apply(x.pType, resultSRVB.end())
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

          val LWORKAddress = mb.newLocal[Long]()

          val M = shapeArray(0)
          val N = shapeArray(1)
          val K = new Value[Long] {
            def get: Code[Long] = (M < N).mux(M, N)
          }
          val LDA = new Value[Long] {
            override def get: Code[Long] = (M > 1L).mux(M, 1L) // Possible stride tricks could change this in the future.
          }

          def LWORK = Region.loadDouble(LWORKAddress).toI

          val dataAddress = pndValue.pt.data.load(pndValue.tcode[Long])

          val tauPType = PCanonicalArray(PFloat64Required, true)
          val tauAddress = cb.newLocal[Long]("ndarray_qr_tauAddress")
          val workAddress = cb.newLocal[Long]("ndarray_qr_workAddress")
          val aAddressDGEQRF = cb.newLocal[Long]("ndarray_qr_aAddressDGEQRF") // Should be column major
          val rDataAddress = cb.newLocal[Long]("ndarray_qr_rDataAddress")
          val aNumElements = cb.newLocal[Long]("ndarray_qr_aNumElements")

          val infoDGEQRFResult = cb.newLocal[Int]("ndaray_qr_infoDGEQRFResult")
          val infoDGEQRFErrorTest = (extraErrorMsg: String) => (infoDGEQRFResult cne 0)
            .orEmpty(Code._fatal[Unit](const(s"LAPACK error DGEQRF. $extraErrorMsg Error code = ").concat(infoDGEQRFResult.toS)))

          // Computing H and Tau
          cb.assign(aNumElements, pndValue.pt.numElements(shapeArray, mb))
          cb.assign(aAddressDGEQRF, pndValue.pt.data.pType.allocate(region.code, aNumElements.toI))
          cb.append(pndValue.pt.data.pType.stagedInitialize(aAddressDGEQRF, aNumElements.toI))
          cb.append(Region.copyFrom(pndValue.pt.data.pType.firstElementOffset(dataAddress, (M * N).toI),
            pndValue.pt.data.pType.firstElementOffset(aAddressDGEQRF, aNumElements.toI), (M * N) * 8L))

          cb.assign(tauAddress, tauPType.allocate(region.code, K.toI))
          cb.append(tauPType.stagedInitialize(tauAddress, K.toI))

          cb.assign(LWORKAddress, region.code.allocate(8L, 8L))

          cb.assign(infoDGEQRFResult, Code.invokeScalaObject7[Int, Int, Long, Int, Long, Long, Int, Int](LAPACK.getClass, "dgeqrf",
            M.toI,
            N.toI,
            pndValue.pt.data.pType.firstElementOffset(aAddressDGEQRF, aNumElements.toI),
            LDA.toI,
            tauPType.firstElementOffset(tauAddress, K.toI),
            LWORKAddress,
            -1
          ))
          cb.append(infoDGEQRFErrorTest("Failed size query."))

          cb.assign(workAddress, Code.invokeStatic1[Memory, Long, Long]("malloc", LWORK.toL * 8L))
          cb.assign(infoDGEQRFResult, Code.invokeScalaObject7[Int, Int, Long, Int, Long, Long, Int, Int](LAPACK.getClass, "dgeqrf",
            M.toI,
            N.toI,
            pndValue.pt.data.pType.firstElementOffset(aAddressDGEQRF, aNumElements.toI),
            LDA.toI,
            tauPType.firstElementOffset(tauAddress, K.toI),
            workAddress,
            LWORK
          ))
          cb.append(Code.invokeStatic1[Memory, Long, Unit]("free", workAddress.load()))
          cb.append(infoDGEQRFErrorTest("Failed to compute H and Tau."))

          val result: Code[Long] = if (mode == "raw") {
            val rawPType = x.pType.asInstanceOf[PTuple]
            val rawOutputSrvb = new StagedRegionValueBuilder(mb, x.pType, region.code)
            val hPType = rawPType.types(0).asInstanceOf[PNDArray]
            val tauPType = rawPType.types(1).asInstanceOf[PNDArray]

            val hShapeArray = FastIndexedSeq[Value[Long]](N, M)
            val hShapeBuilder = hPType.makeShapeBuilder(hShapeArray)
            val hStridesBuilder = hPType.makeRowMajorStridesBuilder(hShapeArray, mb)

            val tauShapeBuilder = tauPType.makeShapeBuilder(FastIndexedSeq(K))
            val tauStridesBuilder = tauPType.makeRowMajorStridesBuilder(FastIndexedSeq(K), mb)

            val h = hPType.construct(hShapeBuilder, hStridesBuilder, aAddressDGEQRF, mb, region.code)
            val tau = tauPType.construct(tauShapeBuilder, tauStridesBuilder, tauAddress, mb, region.code)

            val constructHAndTauTuple = Code(
              rawOutputSrvb.start(),
              rawOutputSrvb.addIRIntermediate(h),
              rawOutputSrvb.advance(),
              rawOutputSrvb.addIRIntermediate(tau),
              rawOutputSrvb.advance(),
              rawOutputSrvb.end()
            )

            constructHAndTauTuple

          } else {
            val (rPType, rRows, rCols) = if (mode == "r") {
              (x.pType.asInstanceOf[PNDArray], K, N)
            } else if (mode == "complete") {
              (x.pType.asInstanceOf[PTuple].types(1).asInstanceOf[PNDArray], M, N)
            } else if (mode == "reduced") {
              (x.pType.asInstanceOf[PTuple].types(1).asInstanceOf[PNDArray], K, N)
            } else {
              throw new AssertionError(s"Unsupported QR mode $mode")
            }

            val rShapeArray = FastIndexedSeq[Value[Long]](rRows, rCols)

            val rShapeBuilder = rPType.makeShapeBuilder(rShapeArray)
            val rStridesBuilder = rPType.makeColumnMajorStridesBuilder(rShapeArray, mb)

            cb.assign(rDataAddress, rPType.data.pType.allocate(region.code, aNumElements.toI))
            cb.append(rPType.data.pType.stagedInitialize(rDataAddress, (rRows * rCols).toI))

            // This block assumes that `rDataAddress` and `aAddressDGEQRF` point to column major arrays.
            // TODO: Abstract this into ndarray ptype/pcode interface methods.
            val currRow = cb.newLocal[Int]("ndarray_qr_currRow")
            val currCol = cb.newLocal[Int]("ndarray_qr_currCol")

            cb.forLoop({cb.assign(currCol, 0)}, currCol < rCols.toI, {cb.assign(currCol, currCol + 1)}, {
              cb.forLoop({cb.assign(currRow, 0)}, currRow < rRows.toI, {cb.assign(currRow, currRow + 1)}, {
                cb.append(Region.storeDouble(
                  pndValue.pt.data.pType.elementOffset(rDataAddress, aNumElements.toI, currCol * rRows.toI + currRow),
                  (currCol >= currRow).mux(
                    Region.loadDouble(pndValue.pt.data.pType.elementOffset(aAddressDGEQRF, aNumElements.toI, currCol * M.toI + currRow)),
                    0.0
                  )
                ))
              })
            })

            val computeR = rPType.construct(rShapeBuilder, rStridesBuilder, rDataAddress, mb, region.code)

            if (mode == "r") {
              computeR.tcode[Long]
            }
            else {
              val crPType = x.pType.asInstanceOf[PTuple]
              val crOutputSrvb = new StagedRegionValueBuilder(mb, crPType, region.code)

              val qPType = crPType.types(0).asInstanceOf[PNDArray]
              val qShapeArray = if (mode == "complete") Array(M, M) else Array(M, K)
              val qShapeBuilder = qPType.makeShapeBuilder(qShapeArray)
              val qStridesBuilder = qPType.makeColumnMajorStridesBuilder(qShapeArray, mb)

              val qDataAddress = cb.newLocal[Long]("ndarray_qr_qDataAddress")

              val infoDORGQRResult = cb.newLocal[Int]("ndarray_qr_DORGQR_info")
              val infoDORQRErrorTest = (extraErrorMsg: String) => (infoDORGQRResult cne 0)
                .orEmpty(Code._fatal[Unit](const(s"LAPACK error DORGQR. $extraErrorMsg Error code = ").concat(infoDORGQRResult.toS)))

              val qCondition = cb.newLocal[Boolean]("ndarray_qr_qCondition")
              val numColsToUse = cb.newLocal[Long]("ndarray_qr_numColsToUse")
              val aAddressDORGQR = cb.newLocal[Long]("ndarray_qr_dorgqr_a")

              val qNumElements = cb.newLocal[Long]("ndarray_qr_qNumElements")


              val rNDArray = computeR.memoize(cb, "ndarray_qr_rNDAArrayAddress")
              cb.assign(qCondition, const(mode == "complete") && (M > N))
              cb.assign(numColsToUse, qCondition.mux(M, K))
              cb.assign(qNumElements, M * numColsToUse)

              cb.ifx(qCondition, {
                cb.assign(aAddressDORGQR, pndValue.pt.data.pType.allocate(region.code, qNumElements.toI))
                cb.append(qPType.data.pType.stagedInitialize(aAddressDORGQR, qNumElements.toI))
                cb.append(Region.copyFrom(pndValue.pt.data.pType.firstElementOffset(aAddressDGEQRF, aNumElements.toI),
                  qPType.data.pType.firstElementOffset(aAddressDORGQR, qNumElements.toI), aNumElements * 8L))
              }, {
                cb.assign(aAddressDORGQR, aAddressDGEQRF)
              })

              cb.assign(infoDORGQRResult, Code.invokeScalaObject8[Int, Int, Int, Long, Int, Long, Long, Int, Int](LAPACK.getClass, "dorgqr",
                M.toI,
                numColsToUse.toI,
                K.toI,
                pndValue.pt.data.pType.firstElementOffset(aAddressDORGQR, aNumElements.toI),
                LDA.toI,
                tauPType.firstElementOffset(tauAddress, K.toI),
                LWORKAddress,
                -1
              ))
              cb.append(infoDORQRErrorTest("Failed size query."))
              cb.append(workAddress := Code.invokeStatic1[Memory, Long, Long]("malloc", LWORK.toL * 8L))
              cb.assign(infoDORGQRResult, Code.invokeScalaObject8[Int, Int, Int, Long, Int, Long, Long, Int, Int](LAPACK.getClass, "dorgqr",
                M.toI,
                numColsToUse.toI,
                K.toI,
                pndValue.pt.data.pType.elementOffset(aAddressDORGQR, (M * numColsToUse).toI, 0),
                LDA.toI,
                tauPType.elementOffset(tauAddress, K.toI, 0),
                workAddress,
                LWORK
              ))
              cb.append(Code.invokeStatic1[Memory, Long, Unit]("free", workAddress.load()))
              cb.append(infoDORQRErrorTest("Failed to compute Q."))
              cb.assign(qDataAddress, qPType.data.pType.allocate(region.code, qNumElements.toI))
              cb.append(qPType.data.pType.stagedInitialize(qDataAddress, qNumElements.toI))
              cb.append(Region.copyFrom(pndValue.pt.data.pType.firstElementOffset(aAddressDORGQR),
                qPType.data.pType.firstElementOffset(qDataAddress), (M * numColsToUse) * 8L))

              val computeCompleteOrReduced = Code(Code(FastIndexedSeq(
                crOutputSrvb.start(),
                crOutputSrvb.addIRIntermediate(qPType.construct(qShapeBuilder, qStridesBuilder, qDataAddress, mb, region.code)),
                crOutputSrvb.advance(),
                crOutputSrvb.addIRIntermediate(rNDArray),
                crOutputSrvb.advance())),
                crOutputSrvb.end()
              )

              computeCompleteOrReduced
            }
          }
          PCode(pt, result)
        }
      case x: NDArrayMap  =>  emitDeforestedNDArray(x)
      case x: NDArrayMap2 =>  emitDeforestedNDArray(x)
      case x: NDArrayReshape => emitDeforestedNDArray(x)
      case x: NDArrayConcat => emitDeforestedNDArray(x)
      case x: NDArraySlice => emitDeforestedNDArray(x)
      case x: NDArrayFilter => emitDeforestedNDArray(x)
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
        val newRegion = cb.newField("resultop_region", region.code)
        val AggContainer(aggs, sc, _) = container.get
        val srvb = new StagedRegionValueBuilder(cb.emb, x.pType, newRegion)
        cb += srvb.start()

        (0 until aggs.length).foreach { j =>
          val idx = start + j
          val rvAgg = agg.Extract.getAgg(sig(j))
          rvAgg.result(cb, sc.states(idx), srvb)
          cb += srvb.advance()
        }

        presentC(srvb.offset)

      case x@ApplySeeded(fn, args, seed, rt) =>
        val codeArgs = args.map(a => (a.pType, () => emitI(a)))
        val impl = x.implementation
        val unified = impl.unify(Array.empty[Type], args.map(_.typ), rt)
        assert(unified)
        impl.applySeededI(seed, cb, EmitRegion(mb, region.code), pt, codeArgs: _*)

      case AggStateValue(i, _) =>
        val AggContainer(_, sc, _) = container.get
        presentC(sc.states(i).serializeToRegion(cb, coerce[PBinary](pt), region.code))

      case x@StreamFold(a, zero, accumName, valueName, body) =>
        val streamType = coerce[PStream](a.pType)
        val eltType = streamType.elementType
        val accType = x.accPType
        val outerRegion = region.asParent(streamType.separateRegions, "StreamFold")
        val eltRegion = outerRegion.createChildRegion(mb)
        val tmpRegion = outerRegion.createChildRegion(mb)

        val streamOpt = emitStream(a, outerRegion)
        streamOpt.flatMap(cb) { stream =>
          val xAcc = mb.newEmitField(accumName, accType)
          val xElt = mb.newEmitField(valueName, eltType)

          cb += eltRegion.allocateRegion(Region.REGULAR)
          cb += tmpRegion.allocateRegion(Region.REGULAR)
          cb.assign(xAcc, emitI(zero, eltRegion).map(cb)(_.castTo(cb, eltRegion.code, accType)))

          stream.asStream.stream.getStream(eltRegion).forEachI(ctx, cb, { elt =>
            // pre- and post-condition: 'xAcc' contains current accumulator,
            // whose heap memory is contained in 'eltRegion'. 'tmpRegion' is
            // empty.
            cb.assign(xElt, elt)
            cb.assign(xAcc, emitI(body, eltRegion, env.bind(accumName -> xAcc, valueName -> xElt))
              .map(cb)(eltRegion.copyTo(cb, _, tmpRegion, accType)))
            cb += eltRegion.clear()
            cb += StagedRegion.swap(mb, eltRegion, tmpRegion)
          })

          cb += tmpRegion.free()
          cb.assign(xAcc, xAcc.map(eltRegion.copyToParent(cb, _)))
          cb += eltRegion.free()

          xAcc.toI(cb)
        }

      case x@StreamFold2(a, acc, valueName, seq, res) =>
        val streamType = coerce[PStream](a.pType)
        val eltType = streamType.elementType

        val xElt = mb.newEmitField(valueName, eltType)
        val names = acc.map(_._1)
        val accTypes = x.accPTypes
        val accVars = (names, accTypes).zipped.map(mb.newEmitField)
        val tmpAccVars = (names, accTypes).zipped.map(mb.newEmitField)

        val outerRegion = region.asParent(streamType.separateRegions, "StreamFold2")
        val eltRegion = outerRegion.createChildRegion(mb)
        val tmpRegion = outerRegion.createChildRegion(mb)

        val resEnv = env.bind(names.zip(accVars): _*)
        val seqEnv = resEnv.bind(valueName, xElt)

        val streamOpt = emitStream(a, outerRegion)
        streamOpt.flatMap(cb) { stream =>
          cb += eltRegion.allocateRegion(Region.REGULAR)
          cb += tmpRegion.allocateRegion(Region.REGULAR)

          (accVars, acc).zipped.foreach { case (xAcc, (_, x)) =>
            cb.assign(xAcc, emitI(x, eltRegion).map(cb)(_.castTo(cb, eltRegion.code, xAcc.pt)))
          }
          stream.asStream.stream.getStream(eltRegion).forEachI(ctx, cb, { elt =>
            // pre- and post-condition: 'accVars' contain current accumulators,
            // all of whose heap memory is contained in 'eltRegion'. 'tmpRegion'
            // is empty.
            cb.assign(xElt, elt)
            (tmpAccVars, seq).zipped.foreach { (accVar, ir) =>
              cb.assign(accVar,
                emitI(ir, eltRegion, env = seqEnv)
                  .map(cb)(eltRegion.copyTo(cb, _, tmpRegion, accVar.pt)))
            }
            (accVars, tmpAccVars).zipped.foreach { (v, tmp) => cb.assign(v, tmp) }
            cb += eltRegion.clear()
            StagedRegion.swap(mb, eltRegion, tmpRegion)
          })
          cb += tmpRegion.free()
          accVars.foreach { xAcc =>
            cb.assign(xAcc, xAcc.map(eltRegion.copyToParent(cb, _)))
          }
          cb += eltRegion.free()

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
          cb, region.code, shufflePType, shuffle.uuid())

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
        val shuffleType = coerce[TShuffle](idIR.typ)
        val rowsPType = coerce[PStream](rowsIR.pType)
        val uuid = emitI(idIR)
          .get(cb, "shuffle ID must be non-missing")
          .asInstanceOf[SCanonicalShufflePointerCode]
          .memoize(cb, "shuffleClientUUID")
        val shuffle = CodeShuffleClient.createValue(
          cb,
          mb.ecb.getType(shuffleType),
          uuid.loadBytes(),
          mb.ecb.getPType(rowsPType.elementType),
          Code._null)
        cb += shuffle.startPut()

        val outerRegion = region.asParent(rowsPType.separateRegions, "ShuffleWrite")
        val eltRegion = outerRegion.createChildRegion(mb)
        val rows = emitStream(rowsIR, outerRegion)
          .get(cb, "rows stream was missing in shuffle write")
          .asStream.stream.getStream(eltRegion)
        cb += eltRegion.allocateRegion(Region.REGULAR)
        cb += rows.forEach(ctx, mb, { row: EmitCode =>
          Code(
            row.setup,
            row.m.mux(
              Code._fatal[Unit]("cannot handle empty rows in shuffle put"),
              Code(shuffle.putValue(row.value[Long]),
                   eltRegion.clear())))
        })
        cb += eltRegion.free()
        cb += shuffle.putValueDone()
        cb += shuffle.endPut()
        cb += shuffle.close()

        val resPType = pt.asInstanceOf[PCanonicalBinary]
        // FIXME: server needs to send uuid for the successful partition
        val boff = cb.memoize(resPType.loadCheapPCode(cb, resPType.allocate(region.code, 0)),"shuffleWriteBOff")
        cb += resPType.storeLength(boff.tcode[Long], 0)
        presentPC(boff)

      case x@ReadValue(path, spec, requestedType) =>
        emitI(path).map(cb) { pv =>
          val ib = cb.newLocal[InputBuffer]("read_ib")
          cb.assign(ib, spec.buildCodeInputBuffer(mb.open(pv.asString.loadString(), checkCodec = true)))
          spec.buildEmitDecoder(requestedType, mb.ecb)(region.code, ib)
        }

      case WriteValue(value, path, spec) =>
        emitI(path).flatMap(cb) { case p: PStringCode =>
          val pv = p.memoize(cb, "write_path")
          emitI(value).map(cb) { v =>
            val ob = cb.newLocal[OutputBuffer]("write_ob")
            cb.assign(ob, spec.buildCodeOutputBuffer(mb.create(pv.asString.loadString())))
            val enc = spec.buildEmitEncoder(v.pt, cb.emb.ecb)
            cb += enc(region.code, v.memoize(cb, "write_value"), ob)
            cb += ob.invoke[Unit]("close")
            pv
          }
        }

      case _ =>
        emitFallback(ir)
    }
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
    val region = StagedRegion(mb.getCodeParam[Region](1))
    emit(ir, mb, region, env, container, None)
  }

  private[ir] def emitWithRegion(ir: IR, mb: EmitMethodBuilder[C], region: StagedRegion, env: E, container: Option[AggContainer]): EmitCode =
    emit(ir, mb, region, env, container, None)

  private def emit(
    ir: IR,
    mb: EmitMethodBuilder[C],
    region: StagedRegion,
    env: E,
    container: Option[AggContainer],
    loopEnv: Option[Env[LoopRef]],
    fallingBackFromEmitI: Boolean = false
  ): EmitCode = {

    def emit(ir: IR, region: StagedRegion = region, env: E = env, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): EmitCode =
      this.emit(ir, mb, region, env, container, loopEnv)

    def emitInMethod(ir: IR, mb: EmitMethodBuilder[C]): EmitCode =
      this.emit(ir, mb, Env.empty, container)

    def emitI(ir: IR, cb: EmitCodeBuilder, env: E = env, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): IEmitCode =
      this.emitI(ir, cb, region, env, container, loopEnv)

    def emitVoid(ir: IR, env: E = env, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): Code[Unit] = {
      EmitCodeBuilder.scopedVoid(mb) { cb =>
        this.emitVoid(cb, ir, mb, region, env, container, loopEnv)
      }
    }

    def emitStream(ir: IR, outerRegion: ParentStagedRegion): EmitCode =
      EmitStream.emit(ctx, this, ir, mb, outerRegion, env, container)

    def emitNDArrayColumnMajorStrides(ir: IR): EmitCode = {
      EmitCode.fromI(mb) { cb =>
        emit(ir).toI(cb).map(cb) { case pNDCode: PNDArrayCode =>
          val pNDValue = pNDCode.memoize(cb, "ndarray_column_major_check")
          val isColumnMajor = LinalgCodeUtils.checkColumnMajor(pNDValue, cb)
          val pAnswer = cb.emb.newPField("ndarray_output_column_major", pNDValue.pt)
          cb.ifx(isColumnMajor, {cb.assign(pAnswer, pNDValue) },
            {
              cb.assign(pAnswer, LinalgCodeUtils.createColumnMajorCode(pNDValue, cb, region.code))
            })
          pAnswer
        }
      }
    }

    val pt = ir.pType

    // ideally, emit would not be called with void values, but initOp args can be void
    // working towards removing this
    if (pt == PVoid)
      return new EmitCode(emitVoid(ir), const(false), PCode._empty)

    (ir: @unchecked) match {
      case Coalesce(values) =>
        val mout = mb.newLocal[Boolean]()
        val out = mb.newPLocal(pt)

        def f(i: Int): Code[Unit] = {
          if (i < values.length) {
            val ec = emit(values(i))
            Code(ec.setup,
              ec.m.mux(
                f(i + 1),
                Code(mout := false, EmitCodeBuilder.scopedVoid(mb){ cb => cb.assign(out, ec.pv.castTo(cb, region.code, pt)) })))
          } else
            mout := true
        }

        EmitCode(
          setup = f(0),
          m = mout,
          pv = out.get)

      case If(cond, cnsq, altr) =>
        assert(cnsq.typ == altr.typ)

        val codeCond = emit(cond)
        val out = mb.newPLocal(pt)
        val mout = mb.newLocal[Boolean]()
        val codeCnsq = emit(cnsq)
        val codeAltr = emit(altr)

        val setup = Code(
          codeCond.setup,
          codeCond.m.mux(
            mout := true,
            coerce[Boolean](codeCond.v).mux(
              Code(codeCnsq.setup,
                mout := codeCnsq.m,
                mout.mux(
                  Code._empty,
                  EmitCodeBuilder.scopedVoid(mb) { cb => cb.assign(out, codeCnsq.pv.castTo(cb, region.code, ir.pType)) })),
              Code(codeAltr.setup,
                mout := codeAltr.m,
                mout.mux(
                  Code._empty,
                  EmitCodeBuilder.scopedVoid(mb) { cb => cb.assign(out, codeAltr.pv.castTo(cb, region.code, ir.pType)) })))))

        EmitCode(setup, mout, out.load())

      case Let(name, value, body) => value.pType match {
        case streamType: PCanonicalStream =>
          val outerRegion = region.asParent(streamType.separateRegions, "Let value")
          val valuet = emitStream(value, outerRegion)
          val bodyEnv = env.bind(name -> new EmitUnrealizableValue(streamType, valuet))

          emit(body, env = bodyEnv)

        case valueType =>
          val x = mb.newEmitField(name, valueType)
          EmitCodeBuilder.scopedEmitCode(mb) { cb =>
            x.store(cb, emit(value))
            val bodyenv = env.bind(name, x)
            emit(body, env = bodyenv)
          }
      }

      case Ref(name, _) =>
        val ev = env.lookup(name)
        if (ev.pt != pt)
          throw new RuntimeException(s"PValue type did not match inferred ptype:\n name: $name\n  pv: ${ ev.pt }\n  ir: $pt")
        ev.get

      case x@MakeArray(args, _) =>
        val pType = x.pType.asInstanceOf[PArray]
        val srvb = new StagedRegionValueBuilder(mb, pType, region.code)

        val addElts = args.map { arg =>
          val v = emit(arg)
          Code(
            v.setup,
            v.m.mux(srvb.setMissing(), srvb.addIRIntermediate(v.pv, deepCopy = false)),
            srvb.advance())
        }
        present(pt, Code(srvb.start(args.size, init = true),
          Code(addElts),
          srvb.offset))

      case x@(_: ArraySort | _: ToSet | _: ToDict) =>
        val resultTypeAsIterable = coerce[PIterable](x.pType)
        val eltType = x.children(0).asInstanceOf[IR].pType.asInstanceOf[PIterable].elementType
        val eltVType = eltType.virtualType

        val vab = new StagedArrayBuilder(resultTypeAsIterable.elementType, mb, 0)
        val sorter = new ArraySorter(EmitRegion(mb, region.code), vab)

        val (array, lessThan, distinct, leftRightComparatorNames: Array[String]) = (x: @unchecked) match {
          case ArraySort(a, l, r, lessThan) => (a, lessThan, Code._empty, Array(l, r))
          case ToSet(a) =>
            val discardNext = mb.genEmitMethod("discardNext",
              FastIndexedSeq[ParamType](typeInfo[Region], eltType.asEmitParam, eltType.asEmitParam),
              typeInfo[Boolean])
            val cmp2 = ApplyComparisonOp(EQWithNA(eltVType), In(0, eltType), In(1, eltType))
            InferPType(cmp2)
            val EmitCode(s, m, pv) = emitInMethod(cmp2, discardNext)
            discardNext.emit(Code(s, m || pv.tcode[Boolean]))
            val lessThan = ApplyComparisonOp(Compare(eltVType), In(0, eltType), In(1, eltType)) < 0
            InferPType(lessThan)
            (a, lessThan, sorter.distinctFromSorted { (r, v1, m1, v2, m2) =>
              EmitCodeBuilder.scopedCode[Boolean](mb) { cb =>
                cb.invokeCode[Boolean](discardNext, r,
                  new EmitCode(Code._empty, m1, PCode(eltType, v1)),
                  new EmitCode(Code._empty, m2, PCode(eltType, v2)))
              }
            }, Array.empty[String])
          case ToDict(a) =>
            val (k0, k1, keyType) = eltType match {
              case t: PStruct => (GetField(In(0, eltType), "key"), GetField(In(1, eltType), "key"), t.fieldType("key"))
              case t: PTuple => (GetTupleElement(In(0, eltType), 0), GetTupleElement(In(1, eltType), 0), t.types(0))
            }
            val discardNext = mb.genEmitMethod("discardNext",
              FastIndexedSeq[ParamType](typeInfo[Region], eltType.asEmitParam, eltType.asEmitParam),
              typeInfo[Boolean])

            val cmp2 = ApplyComparisonOp(EQWithNA(keyType.virtualType), k0, k1).deepCopy()
            InferPType(cmp2)
            val EmitCode(s, m, pv) = emitInMethod(cmp2, discardNext)
            discardNext.emit(Code(s, m || pv.tcode[Boolean]))
            val lessThan = (ApplyComparisonOp(Compare(keyType.virtualType), k0, k1) < 0).deepCopy()
            InferPType(lessThan)
            (a, lessThan, Code(sorter.pruneMissing, sorter.distinctFromSorted { (r, v1, m1, v2, m2) =>
              EmitCodeBuilder.scopedCode[Boolean](mb) { cb =>
                cb.invokeCode[Boolean](discardNext, r,
                  new EmitCode(Code._empty, m1, PCode(eltType, v1)),
                  new EmitCode(Code._empty, m2, PCode(eltType, v2)))
              }
            }), Array.empty[String])
        }

        val sort = vab.ti match {
          case BooleanInfo => sorter.sort(makeDependentSortingFunction[Boolean](
            region.code, eltType, lessThan, env, leftRightComparatorNames))
          case IntInfo => sorter.sort(makeDependentSortingFunction[Int](region.code, eltType, lessThan, env, leftRightComparatorNames))
          case LongInfo => sorter.sort(makeDependentSortingFunction[Long](
            region.code, eltType, lessThan, env, leftRightComparatorNames))
          case FloatInfo => sorter.sort(makeDependentSortingFunction[Float](
            region.code, eltType, lessThan, env, leftRightComparatorNames))
          case DoubleInfo => sorter.sort(makeDependentSortingFunction[Double](
            region.code, eltType, lessThan, env, leftRightComparatorNames))
        }

        val outerRegion = region.asParent(coerce[PStream](array.pType).separateRegions, "ArraySort")
        val optStream = emitStream(array, outerRegion)
        optStream.map { stream =>
          PCode(pt, Code(
            EmitStream.write(ctx, mb, stream.asStream, vab, outerRegion),
            sort,
            distinct,
            sorter.toRegion()))
        }

      case CastToArray(a) =>
        val et = emit(a)
        EmitCode(et.setup, et.m, PCode(pt, et.v))

      case ToArray(a) =>
        val outerRegion = region.asParent(coerce[PStream](a.pType).separateRegions, "ToArray")
        emitStream(a, outerRegion).map { stream =>
          EmitStream.toArray(ctx, mb, coerce[PArray](pt), stream.asStream, outerRegion)
        }

      case x@LowerBoundOnOrderedCollection(orderedCollection, elem, onKey) =>
        val typ: PContainer = coerce[PIterable](orderedCollection.pType).asPContainer
        val a = emit(orderedCollection)
        val e = emit(elem)
        val bs = new BinarySearch[C](mb, typ, elem.pType, keyOnly = onKey)

        val localA = mb.newLocal[Long]()
        val localElementMB = mb.newLocal[Boolean]()
        val localElementValue = mb.newLocal()(typeToTypeInfo(elem.pType))
        EmitCode(
          Code(a.setup, e.setup),
          a.m,
          PCode(pt, Code(
            localA := a.value[Long],
            localElementMB := e.m,
            localElementMB.mux(Code._empty, localElementValue.storeAny(e.v)),
            bs.getClosestIndex(localA, localElementMB, localElementValue))))

      case GroupByKey(collection) =>
        // sort collection by group
        val atyp = coerce[PStream](collection.pType)
        val etyp = coerce[PBaseStruct](atyp.elementType)
        val ktyp = etyp.types(0)
        val vtyp = etyp.types(1)
        val eltOut = coerce[PDict](ir.pType).elementType

        val eab = new StagedArrayBuilder(etyp, mb, 16)
        val sorter = new ArraySorter(EmitRegion(mb, region.code), eab)

        val (k1, k2) = etyp match {
          case t: PStruct => GetField(In(0, t), "key") -> GetField(In(1, t), "key")
          case t: PTuple =>
            assert(t.fields(0).index == 0)
            GetTupleElement(In(0, t), 0) -> GetTupleElement(In(1, t), 0)
        }

        val compare = ApplyComparisonOp(Compare(etyp.types(0).virtualType), k1, k2) < 0
        InferPType(compare)
        val leftRightComparatorNames = Array.empty[String]
        val sortF = eab.ti match {
          case BooleanInfo => makeDependentSortingFunction[Boolean](region.code, etyp, compare, env, leftRightComparatorNames)
          case IntInfo => makeDependentSortingFunction[Int](region.code, etyp, compare, env, leftRightComparatorNames)
          case LongInfo => makeDependentSortingFunction[Long](region.code, etyp, compare, env, leftRightComparatorNames)
          case FloatInfo => makeDependentSortingFunction[Float](region.code, etyp, compare, env, leftRightComparatorNames)
          case DoubleInfo => makeDependentSortingFunction[Double](region.code, etyp, compare, env, leftRightComparatorNames)
        }

        val nab = new StagedArrayBuilder(PInt32(), mb, 0)
        val i = mb.newLocal[Int]()

        def loadKey(n: Code[Int]): Code[_] =
          Region.loadIRIntermediate(ktyp)(etyp.fieldOffset(coerce[Long](eab(n)), 0))

        def loadValue(n: Code[Int]): Code[_] =
          Region.loadIRIntermediate(vtyp)(etyp.fieldOffset(coerce[Long](eab(n)), 1))

        val srvb = new StagedRegionValueBuilder(mb, ir.pType, region.code)

        type E = Env[(TypeInfo[_], Code[Boolean], Code[_])]

        val (lastKey, currKey) = (etyp.virtualType: @unchecked) match {
          case ts: TStruct =>
            GetField(In(0, etyp), ts.fieldNames(0)) -> GetField(In(1, etyp), ts.fieldNames(0))
          case tt: TTuple =>
            GetTupleElement(In(0, etyp), tt.fields(0).index) -> GetTupleElement(In(1, etyp), tt.fields(0).index)
        }
        val compare2 = ApplyComparisonOp(EQWithNA(ktyp.virtualType), lastKey, currKey)
        InferPType(compare2)
        val isSame = mb.genEmitMethod("isSame",
          FastIndexedSeq(typeInfo[Region], etyp.asEmitParam, etyp.asEmitParam),
          BooleanInfo)
        isSame.emitWithBuilder { cb =>
          val isSameCode = emitInMethod(compare2, isSame)
          cb += isSameCode.setup
          isSameCode.m || isSameCode.value[Boolean]
        }

        val outerRegion = region.asParent(atyp.separateRegions, "GroupByKey")
        emitStream(collection, outerRegion).map { stream =>
          PCode(pt, Code(
            EmitStream.write(ctx, mb, stream.asStream, eab, outerRegion),
            sorter.sort(sortF),
            sorter.pruneMissing,
            eab.size.ceq(0).mux(
              Code(srvb.start(0), srvb.offset),
              Code(
                nab.clear,
                i := 1,
                nab.add(1),
                Code.whileLoop(i < eab.size,
                  EmitCodeBuilder.scopedCode[Boolean](mb) { cb =>
                    cb.invokeCode[Boolean](isSame, region.code,
                      eab.applyEV(mb, i-1), eab.applyEV(mb, i))
                  }.mux(
                    nab.update(nab.size - 1, coerce[Int](nab(nab.size - 1)) + 1),
                    nab.add(1)),
                  i += 1),
                i := 0,
                srvb.start(nab.size),
                Code.whileLoop(srvb.arrayIdx < nab.size,
                  srvb.addBaseStruct(eltOut, { structbuilder =>
                    Code(
                      structbuilder.start(),
                      structbuilder.addIRIntermediate(ktyp)(loadKey(i)),
                      structbuilder.advance(),
                      structbuilder.addArray(coerce[PArray](eltOut.types(1)), { arraybuilder =>
                        Code(
                          arraybuilder.start(coerce[Int](nab(srvb.arrayIdx))),
                          Code.whileLoop(arraybuilder.arrayIdx < coerce[Int](nab(srvb.arrayIdx)),
                            etyp.isFieldMissing(coerce[Long](eab(i)), 1).mux(
                              arraybuilder.setMissing(),
                              arraybuilder.addIRIntermediate(etyp.types(1))(loadValue(i))
                              ),
                            i += 1,
                            arraybuilder.advance()
                            ))
                      }))
                  }),
                  srvb.advance()),
                srvb.offset))))
        }

      case ArrayZeros(length) =>
        val lengthTriplet = emit(length)
        val outputPType = coerce[PArray](ir.pType)
        val elementSize = outputPType.elementByteSize
        val numElements = mb.genFieldThisRef[Int]()
        val arrayAddress = mb.genFieldThisRef[Long]()
        val result = Code(
          numElements := lengthTriplet.value[Int],
          arrayAddress := outputPType.allocate(region.code, numElements),
          outputPType.stagedInitialize(arrayAddress, numElements),
          Region.setMemory(outputPType.firstElementOffset(arrayAddress), numElements.toL * elementSize, 0.toByte),
          arrayAddress
        )
        EmitCode(lengthTriplet.setup, lengthTriplet.m, PCode(pt, result))

      case x@StreamLen(a) =>
        val outerRegion = region.asParent(coerce[PStream](a.pType).separateRegions, "StreamLen")
        emitStream(a, outerRegion).map { ss =>
          val count = mb.newLocal[Int]("stream_length")
          val SizedStream(setup, stream, length) = ss.asStream.stream
          val lenCode =
            length match {
              case Some(len) => Code(setup, len)
              case None =>
                val eltRegion = outerRegion.createChildRegion(mb)
                Code(
                  count := 0,
                  setup,
                  eltRegion.allocateRegion(Region.REGULAR),
                  stream(eltRegion).forEach(ctx, mb, _ => Code(count := count + 1, eltRegion.clear())),
                  eltRegion.free(),
                  count.get
                )
            }
          PCode(x.pType, lenCode)
        }

      case x@MakeStruct(fields) =>
        val srvb = new StagedRegionValueBuilder(mb, x.pType, region.code)
        val addFields = fields.map { case (_, x) =>
          val v = emit(x)
          Code(
            v.setup,
            v.m.mux(srvb.setMissing(), srvb.addIRIntermediate(x.pType)(v.v)),
            srvb.advance())
        }
        present(pt, Code(srvb.start(init = true), Code(addFields), srvb.offset))

      case x@SelectFields(oldStruct, fields) =>
        val old = emit(oldStruct)

        if(HailContext.getFlag("use_spicy_ptypes") != null) {
          EmitCode(
            old.setup,
            old.m,
            PCode(pt, old.value[Long])
          )
        } else {
          val oldt = coerce[PStruct](oldStruct.pType)
          val oldv = mb.genFieldThisRef[Long]()
          val srvb = new StagedRegionValueBuilder(mb, x.pType, region.code)

          val addFields = fields.map { name =>
            val i = oldt.fieldIdx(name)
            val t = oldt.types(i)
            val fieldMissing = oldt.isFieldMissing(oldv, i)
            val fieldValue = Region.loadIRIntermediate(t)(oldt.fieldOffset(oldv, i))
            Code(
              fieldMissing.mux(
                srvb.setMissing(),
                srvb.addIRIntermediate(t)(fieldValue)),
              srvb.advance())
          }

          EmitCode(
            old.setup,
            old.m,
            PCode(pt, Code(
              oldv := old.value[Long],
              srvb.start(),
              Code(addFields),
              srvb.offset)))
        }

      case x@InsertFields(old, fields, fieldOrder) =>
        if (fields.isEmpty)
          emit(old)
        else
          old.pType match {
            case oldtype: PStruct =>
              val codeOld = emit(old)
              val xo = mb.genFieldThisRef[Long]()
              val updateMap = Map(fields: _*)
              val srvb = new StagedRegionValueBuilder(mb, x.pType, region.code)

              def addFields(t: PType, v: EmitCode): Code[Unit] = Code(
                v.setup,
                v.m.mux(srvb.setMissing(), srvb.addIRIntermediate(t)(v.v)),
                srvb.advance())

              val items = x.pType.fields.map { f =>
                updateMap.get(f.name) match {
                  case Some(vir) =>
                    addFields(vir.pType, emit(vir))
                  case None =>
                    val oldField = oldtype.field(f.name)
                    Code(
                      oldtype.isFieldMissing(xo, oldField.index).mux(
                        srvb.setMissing(),
                        srvb.addIRIntermediate(f.typ)(Region.loadIRIntermediate(oldField.typ)(oldtype.fieldOffset(xo, oldField.index)))),
                      srvb.advance())
                }
              }

              EmitCode(
                codeOld.setup,
                codeOld.m,
                PCode(pt, Code(
                  srvb.start(init = true),
                  xo := coerce[Long](codeOld.v),
                  Code(items),
                  srvb.offset)))
            case _ =>
              val newIR = MakeStruct(fields)
              emit(newIR)
          }

      case x@MakeTuple(fields) =>
        val srvb = new StagedRegionValueBuilder(mb, x.pType, region.code)
        val addFields = fields.map { case (_, x) =>
          val v = emit(x)
          Code(
            v.setup,
            v.m.mux(srvb.setMissing(), srvb.addIRIntermediate(x.pType)(v.v)),
            srvb.advance())
        }
        present(pt, Code(srvb.start(init = true), Code(addFields), srvb.offset))

      case In(i, expectedPType) =>
        // this, Code[Region], ...
        val ev = mb.getEmitParam(2 + i)
        assert(ev.pt == expectedPType)
        ev
      case Die(m, typ, errorId) =>
        val cm = emit(m)
        EmitCode(
          Code(
            cm.setup,
            Code._throw[HailException, Unit](Code.newInstance[HailException, String, Int](
              cm.m.mux[String](
                "<exception message missing>",
                coerce[String](StringFunctions.wrapArg(EmitRegion(mb, region.code), m.pType)(cm.v))), errorId))),
          true,
          pt.defaultValue)

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
        val codeArgs = args.map(emit(_))
        val vars = args.map { a => coerce[Any](mb.newLocal()(typeToTypeInfo(a.pType))) }
        val ins = vars.zip(codeArgs.map(_.v)).map { case (l, i) => l := i }
        val value = Code(Code(ins), meth.invokeCode[Any](
          (CodeParam(region.code) +: vars.map(_.get: Param)): _*))
        strict(pt, value, codeArgs: _*)
      case x@ApplySpecial(_, typeArgs, args, rt) =>
        val codeArgs = args.map(a => emit(a))
        val impl = x.implementation
        val unified = impl.unify(typeArgs, args.map(_.typ), rt)
        assert(unified)
        impl.apply(EmitRegion(mb, region.code), pt, typeArgs, codeArgs: _*)
      case x@CollectDistributedArray(contexts, globals, cname, gname, body, tsd) =>
        val ctxsType = coerce[PStream](contexts.pType)
        val ctxType = ctxsType.elementType
        val gType = globals.pType

        val parentCB = mb.ecb

        val functionID: String = {
          val bodyFB = EmitFunctionBuilder[Region, Array[Byte], Array[Byte], Array[Byte]](ctx, "collect_distributed_array")
          val bodyMB = bodyFB.genEmitMethod("cdaBody",
            Array[ParamType](typeInfo[Region], ctxType.asEmitParam, gType.asEmitParam),
            typeInfo[Long])

          val (cRetPtype, cDec) = x.contextSpec.buildEmitDecoderF[Long](bodyFB.ecb)
          assert(cRetPtype == x.decodedContextPTuple)
          val (gRetPtype, gDec) = x.globalSpec.buildEmitDecoderF[Long](bodyFB.ecb)
          assert(gRetPtype == x.decodedGlobalPTuple)
          val bEnc = x.bodySpec.buildTypedEmitEncoderF[Long](x.bodyPTuple, bodyFB.ecb)
          val bOB = bodyFB.genFieldThisRef[OutputBuffer]()

          val env = Env[EmitValue](
            (cname, bodyMB.getEmitParam(2)),
            (gname, bodyMB.getEmitParam(3)))

          // FIXME fix number of aggs here
          val m = MakeTuple.ordered(FastSeq(body))
          m._pType = PCanonicalTuple(true, body.pType)
          val t = new Emit(ctx, bodyFB.ecb).emit(m, bodyMB, env, None)
          bodyMB.emit(Code(t.setup, t.m.mux(Code._fatal[Long]("return cannot be missing"), t.v)))

          bodyFB.emitWithBuilder { cb =>
            val ctxIB = cb.newLocal[InputBuffer]("cda_ctx_ib", x.contextSpec.buildCodeInputBuffer(
              Code.newInstance[ByteArrayInputStream, Array[Byte]](bodyFB.getCodeParam[Array[Byte]](2))))
            val gIB = cb.newLocal[InputBuffer]("cda_g_ib", x.globalSpec.buildCodeInputBuffer(
              Code.newInstance[ByteArrayInputStream, Array[Byte]](bodyFB.getCodeParam[Array[Byte]](3))))

            val ctxOff = cb.newLocal[Long]("cda_ctx_off", cDec(bodyFB.getCodeParam[Region](1), ctxIB))
            val gOff = cb.newLocal[Long]("cda_g_off", gDec(bodyFB.getCodeParam[Region](1), gIB))

            val bOffCode = cb.invokeCode[Long](bodyMB, bodyFB.getCodeParam[Region](1),
              new EmitCode(Code._empty,
                x.decodedContextPTuple.isFieldMissing(ctxOff, 0),
                PCode(ctxType, Region.loadIRIntermediate(ctxType)(x.decodedContextPTuple.fieldOffset(ctxOff, 0)))),
              new EmitCode(Code._empty,
                x.decodedGlobalPTuple.isFieldMissing(gOff, 0),
                PCode(gType, Region.loadIRIntermediate(gType)(x.decodedGlobalPTuple.fieldOffset(gOff, 0)))))
            val bOff = cb.newLocal[Long]("cda_boff", bOffCode)
            val bOS = cb.newLocal[ByteArrayOutputStream]("cda_baos", Code.newInstance[ByteArrayOutputStream]())
            val bOB = cb.newLocal[OutputBuffer]("cda_ob", x.bodySpec.buildCodeOutputBuffer(bOS))
            cb += bEnc(bodyFB.getCodeParam[Region](1), bOff, bOB)
            cb += bOB.invoke[Unit]("flush")
            cb += bOB.invoke[Unit]("close")
            bOS.invoke[Array[Byte]]("toByteArray")
          }

          val fID = genUID()
          parentCB.addModule(fID, bodyFB.resultWithIndex())
          fID
        }

        val spark = parentCB.backend()

        val outerRegion = region.asParent(ctxsType.separateRegions, "CDA")
        val optCtxStream = COption.fromEmitCode(emitStream(contexts, outerRegion))
        val globalsT = emit(globals)

        val cEnc = x.contextSpec.buildTypedEmitEncoderF[Long](x.contextPTuple, parentCB)
        val gEnc = x.globalSpec.buildTypedEmitEncoderF[Long](x.globalPTuple, parentCB)
        val (bRetPType, bDec) = x.bodySpec.buildEmitDecoderF[Long](parentCB)
        assert(bRetPType == x.decodedBodyPTuple)

        val baos = mb.genFieldThisRef[ByteArrayOutputStream]()
        val buf = mb.genFieldThisRef[OutputBuffer]()
        val ctxab = mb.genFieldThisRef[ByteArrayArrayBuilder]()
        val encRes = mb.genFieldThisRef[Array[Array[Byte]]]()

        def etToTuple(et: EmitCode, t: PType): Code[Long] = {
          val srvb = new StagedRegionValueBuilder(mb, PCanonicalTuple(false, t), region.code)
          Code(
            srvb.start(),
            et.setup,
            et.m.mux(
              srvb.setMissing(),
              srvb.addIRIntermediate(t)(et.v)),
            srvb.offset)
        }

        def addContexts(ctxStream: SizedStream): Code[Unit] = ctxStream match {
          case SizedStream(setup, stream, len) =>
            val eltRegion = outerRegion.createChildRegion(mb)
            Code(
              setup,
              ctxab.invoke[Int, Unit]("ensureCapacity", len.getOrElse(16)),
              eltRegion.allocateRegion(Region.REGULAR),
              stream(eltRegion).map(etToTuple(_, ctxType)).forEach(ctx, mb, { offset =>
                Code(
                  baos.invoke[Unit]("reset"),
                  Code.memoize(offset, "cda_add_contexts_addr") { offset =>
                    cEnc(region.code, offset, buf)
                  },
                  eltRegion.clear(),
                  buf.invoke[Unit]("flush"),
                  ctxab.invoke[Array[Byte], Unit]("add", baos.invoke[Array[Byte]]("toByteArray")))
              }),
              eltRegion.free())
          }

        val addGlobals = Code(
          Code.memoize(etToTuple(globalsT, gType), "cda_g") { g =>
            gEnc(region.code, g, buf)
          },
          buf.invoke[Unit]("flush"))

        val decodeResult = {
          val sab = new StagedRegionValueBuilder(mb, x.pType, region.code)
          val bais = Code.newInstance[ByteArrayInputStream, Array[Byte]](encRes(sab.arrayIdx))
          val eltTupled = mb.genFieldThisRef[Long]()
          Code(
            sab.start(encRes.length()),
            Code.whileLoop(sab.arrayIdx < encRes.length(),
              eltTupled := Code.memoize(x.bodySpec.buildCodeInputBuffer(bais), "decode_ib") { ib =>
                bDec(region.code, ib)
              },
              x.decodedBodyPTuple.isFieldMissing(eltTupled, 0).mux(
                sab.setMissing(),
                EmitCodeBuilder.scopedVoid(mb) { cb =>
                  val pv = x.decodedBodyPType.loadCheapPCode(cb, x.decodedBodyPTuple.loadField(eltTupled, 0))
                  cb += sab.addIRIntermediate(pv)
                }),
              sab.advance()),
            sab.end())
        }

        val optRes = optCtxStream.map { ctxStream => PCode(pt, Code(
          baos := Code.newInstance[ByteArrayOutputStream](),
          buf := x.contextSpec.buildCodeOutputBuffer(baos), // TODO: take a closer look at whether we need two codec buffers?
          ctxab := Code.newInstance[ByteArrayArrayBuilder, Int](16),
          addContexts(ctxStream.asStream.stream),
          baos.invoke[Unit]("reset"),
          addGlobals,
          encRes := spark.invoke[BackendContext, String, Array[Array[Byte]], Array[Byte], Option[TableStageDependency], Array[Array[Byte]]](
            "collectDArray",
            mb.getObject(ctx.backendContext),
            functionID,
            ctxab.invoke[Array[Array[Byte]]]("result"),
            baos.invoke[Array[Byte]]("toByteArray"),
            mb.getObject(tsd)),
          decodeResult))
        }

        COption.toEmitCode(ctx, optRes, mb)

      case x@TailLoop(name, args, body) =>
        val label = CodeLabel()
        val inits = args.zip(x.accPTypes)
        val loopRef = LoopRef(mb, label, inits.map { case ((name, _), pt) => (name, pt) })

        val m = mb.genFieldThisRef[Boolean]()
        val v = mb.newPField(x.pType)

        val argEnv = env
          .bind((args.map(_._1), loopRef.loopArgs).zipped.toArray: _*)

        val newLoopEnv = loopEnv.getOrElse(Env.empty)
        val bodyT = emit(body, env = argEnv, loopEnv = Some(newLoopEnv.bind(name, loopRef)))
        val bodyF = EmitCodeBuilder.scopedVoid(mb) { cb =>
          cb.append(bodyT.setup)
          cb.assign(m, bodyT.m)
          cb.ifx(!m, cb.assign(v, bodyT.pv))
        }
        val initArgs = EmitCodeBuilder.scopedVoid(mb) { cb =>
          cb.assign(loopRef.loopArgs, inits.map { case ((_, x), pt) =>
            emit(x).castTo(mb, region.code, pt)
          })
        }

        EmitCode(Code(initArgs, label, bodyF), m, v.load())

      case Recur(name, args, _) =>
        val loopRef = loopEnv.get.lookup(name)

        EmitCodeBuilder.scopedEmitCode(mb) { cb =>
          cb.assign(loopRef.tmpLoopArgs, loopRef.loopTypes.zip(args).map { case (pt, arg) =>
            emit(arg, loopEnv = None).castTo(mb, region.code, pt)
          })
          cb.assign(loopRef.loopArgs, loopRef.tmpLoopArgs.load())
          cb.append(loopRef.L.goto)
          // dead code
          EmitCode.missing(pt)
        }

      case x@WritePartition(stream, pctx, writer) =>
        val ctxCode = emit(pctx)
        val streamType = coerce[PStream](stream.pType)
        val eltType = coerce[PStruct](streamType.elementType)
        val outerRegion = region.asParent(streamType.separateRegions, "WritePartition")
        COption.toEmitCode(ctx,
          COption.fromEmitCode(emitStream(stream, outerRegion)).flatMap { s =>
            COption.fromEmitCode(writer.consumeStream(ctx, ctxCode, eltType, mb, outerRegion, s.asStream.stream))
          }, mb)

      case x =>
        if (fallingBackFromEmitI) {
          fatal(s"ir is not defined in emit or emitI $x")
        }
        EmitCode.fromI(mb) { cb =>
          emitI(ir, cb)
        }
    }
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
           (id, f.newDepEmitField(e.get))
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

    val sort = f.genEmitMethod("sort",
      FastIndexedSeq(typeInfo[Region], elemPType.asEmitParam, elemPType.asEmitParam),
      BooleanInfo)

    if (leftRightComparatorNames.nonEmpty) {
      assert(leftRightComparatorNames.length == 2)
      newEnv = newEnv.bindIterable(
        IndexedSeq(
          (leftRightComparatorNames(0), sort.getEmitParam(2)),
          (leftRightComparatorNames(1), sort.getEmitParam(3))))
    }

    val EmitCode(setup, m, v) = new Emit(ctx, f.ecb).emit(newIR, sort, newEnv, None)

    sort.emit(Code(setup, m.mux(Code._fatal[Boolean]("Result of sorting function cannot be missing."), v.code)))
    f.apply_method.emitWithBuilder(cb => cb.invokeCode[Boolean](sort, fregion,
      new EmitCode(Code._empty, false, PCode(elemPType, f.getCodeParam[T](1))),
      new EmitCode(Code._empty, false, PCode(elemPType, f.getCodeParam[T](2)))))
    f
  }

  private def present(pv: PCode): EmitCode = EmitCode(Code._empty, false, pv)

  private def present(pt: PType, c: Code[_]): EmitCode =
    EmitCode(Code._empty, false, PCode(pt, c))

  private def strict(pt: PType, value: Code[_], args: EmitCode*): EmitCode = {
    EmitCode(
      Code(args.map(_.setup)),
      if (args.isEmpty) false else args.map(_.m).reduce(_ || _),
      PCode(pt, value))
  }

  def deforestNDArray(x0: IR, mb: EmitMethodBuilder[C], region: StagedRegion, env: E): EmitCode = {
    def emit(ir: IR, env: E = env): EmitCode =
      this.emitWithRegion(ir, mb, region, env, None)

    def dEmit(ir: IR, env: E = env): EmitCode = emit(ir, env)

    def deforest(x: IR): NDArrayEmitter[C] = {
      val xType = coerce[PNDArray](x.pType)
      val nDims = xType.nDims

      x match {
        case NDArrayMap(child, elemName, body) =>
          val childP = child.pType.asInstanceOf[PNDArray]
          val elemPType = childP.elementType


          val childEmitter = deforest(child)
          val setup = childEmitter.setupShape

          new NDArrayEmitter[C](childEmitter.nDims, childEmitter.outputShape,
            childP.shape.pType, body.pType, setup, childEmitter.setupMissing, childEmitter.missing) {
            override def outputElement(elemMB: EmitMethodBuilder[C], idxVars: IndexedSeq[Value[Long]]): Code[_] = {
              assert(elemMB == mb)
              val elemRef = elemMB.newPresentEmitField("ndarray_map_element_name", elemPType)
              val bodyEnv = env.bind(elemName, elemRef)
              val bodyt = dEmit(body, bodyEnv)

              EmitCodeBuilder.scopedCode(mb) { cb =>
                cb.assign(elemRef, PCode(elemPType, childEmitter.outputElement(elemMB, idxVars)))
                cb.append(bodyt.setup)
                cb.ifx(bodyt.m, cb._fatal("NDArray map body cannot be missing"))
                bodyt.v
              }
            }
          }
        case NDArrayMap2(lChild, rChild, lName, rName, body) =>
          val lP = coerce[PNDArray](lChild.pType)
          val rP = coerce[PNDArray](rChild.pType)

          val leftChildEmitter = deforest(lChild)
          val rightChildEmitter = deforest(rChild)

          val (newSetupShape, shapeArray) = NDArrayEmitter.unifyShapes2(mb, leftChildEmitter.outputShape, rightChildEmitter.outputShape)

          val setupMissing = Code(leftChildEmitter.setupMissing, rightChildEmitter.setupMissing)
          val setupShape = Code(leftChildEmitter.setupShape, rightChildEmitter.setupShape, newSetupShape)

          new NDArrayEmitter[C](lP.shape.pType.size, shapeArray, lP.shape.pType, body.pType, setupShape, setupMissing, leftChildEmitter.missing || rightChildEmitter.missing) {
            override def outputElement(elemMB: EmitMethodBuilder[C], idxVars: IndexedSeq[Value[Long]]): Code[_] = {
              assert(elemMB == mb)
              val lElemRef = elemMB.newPresentEmitField(lName, lP.elementType)
              val rElemRef = elemMB.newPresentEmitField(rName, rP.elementType)

              val bodyEnv = env.bind(lName, lElemRef)
                .bind(rName, rElemRef)
              val bodyt = dEmit(body, bodyEnv)

              val lIdxVars2 = NDArrayEmitter.zeroBroadcastedDims2(elemMB, idxVars, nDims, leftChildEmitter.outputShape)
              val rIdxVars2 = NDArrayEmitter.zeroBroadcastedDims2(elemMB, idxVars, nDims, rightChildEmitter.outputShape)

              EmitCodeBuilder.scopedCode(mb) { cb =>
                cb.assign(lElemRef, PCode(lP.elementType, leftChildEmitter.outputElement(elemMB, lIdxVars2)))
                cb.assign(rElemRef, PCode(rP.elementType, rightChildEmitter.outputElement(elemMB, rIdxVars2)))
                cb.append(bodyt.setup)
                cb.ifx(bodyt.m, cb._fatal("NDArray map body cannot be missing"))
                bodyt.v
              }
            }
          }

        case x@NDArrayReindex(child, indexExpr) =>
          val childEmitter = deforest(child)
          val childPType = child.pType.asInstanceOf[PNDArray]

          val outputPType = x.pType
          val outputShapePType = outputPType.shape.pType

          val shapeSeq = indexExpr.map { childIndex =>
            if (childIndex < childPType.nDims)
              childEmitter.outputShape(childIndex)
            else
              const(1L)
          }

          new NDArrayEmitter[C](indexExpr.length, shapeSeq, outputShapePType, outputPType.elementType, childEmitter.setupShape, childEmitter.setupMissing, childEmitter.missing) {
            override def outputElement(elemMB: EmitMethodBuilder[C], idxVars: IndexedSeq[Value[Long]]): Code[_] = {
              val concreteIdxsForChild = Array.tabulate(childEmitter.nDims) { childDim =>
                val parentDim = indexExpr.indexOf(childDim)
                idxVars(parentDim)
              }
              childEmitter.outputElement(elemMB, concreteIdxsForChild)
            }
          }

        case x@NDArrayReshape(childND, shape) =>

          // Need to take this shape, which may have a -1 in it, and turn it into a compatible shape if possible.
          def compatibleShape(numElements: Value[Long], requestedShape: IndexedSeq[Value[Long]]): (Code[Unit], IndexedSeq[Value[Long]]) = {
            val hasNegativeOne = mb.newLocal[Boolean]()
            val runningProduct = mb.newLocal[Long]()
            val replacesNegativeOne = mb.newLocal[Long]()
            val tempShapeElement = mb.newLocal[Long]()

            val newShapeVars = requestedShape.indices.map(_ => mb.genFieldThisRef[Long]())

            val setupShape = coerce[Unit](Code(
              hasNegativeOne := false,
              runningProduct := 1L,

              Code.foreach(requestedShape) { requestedShapeElement => Code(
                tempShapeElement := requestedShapeElement,
                (tempShapeElement < 0L).mux(
                  (tempShapeElement ceq -1L).mux(
                    hasNegativeOne.mux(
                      Code._fatal[Unit]("Can't infer shape, more than one -1"),
                      hasNegativeOne := true
                    ),
                    Code._fatal[Unit]("Can't reshape, new shape must contain only nonnegative numbers or -1")),
                  runningProduct := runningProduct * tempShapeElement
                )
              )},

              Code(
                hasNegativeOne.mux(
                  (runningProduct ceq 0L) || (numElements % runningProduct) > 0L,
                  numElements cne runningProduct
                ).orEmpty(Code._fatal[Unit]("Can't reshape since requested shape is incompatible with number of elements")),
                replacesNegativeOne := (runningProduct ceq 0L).mux(0, numElements / runningProduct),
                Code(newShapeVars.zip(requestedShape).map { case (variable, shapeElement) =>
                  variable := (shapeElement ceq -1L).mux(replacesNegativeOne, shapeElement)
                })
              )
            ))

            (setupShape, newShapeVars)
          }

          val childEmitter = deforest(childND)

          val requestedShapet = emit(shape)
          val requestedShapeAddress = mb.genFieldThisRef[Long]()
          val requestedShapePType = coerce[PTuple](shape.pType)
          val requestedShapeTuple = new CodePTuple(requestedShapePType, requestedShapeAddress)
          val requestedShapeArray = (0 until requestedShapePType.size).map(i => requestedShapeTuple[Long](i)).toArray

          val (childShapeCachingCode, childShapeCached) = childEmitter.outputShape.cacheEntries(mb, LongInfo)

          val numElements = mb.genFieldThisRef[Long]()

          val (reshapeSetup, reshapedShapeArray) = compatibleShape(numElements, requestedShapeArray)

          val setupMissing = Code(
            childEmitter.setupMissing,
            requestedShapet.setup
          )

          val setupShape = Code(
            childEmitter.setupShape,
            childShapeCachingCode,
            requestedShapeAddress := requestedShapet.value[Long],
            numElements := coerce[PNDArray](childND.pType).numElements(childShapeCached, mb),
            reshapeSetup
          )

          new NDArrayEmitter[C](reshapedShapeArray.length, reshapedShapeArray, requestedShapePType.setRequired(true).asInstanceOf[PTuple],
            childEmitter.outputElementPType, setupShape, setupMissing, childEmitter.missing || requestedShapet.m) {
            override def outputElement(elemMB: EmitMethodBuilder[C], idxVars: IndexedSeq[Value[Long]]): Code[_] = {
              val storeElementIndex = elemMB.genFieldThisRef[Long]()

              val (newIdxVarsSetup, newIdxVars) = x.pType.unlinearizeIndexRowMajor(storeElementIndex, childShapeCached, elemMB)

              assert(newIdxVars.length == childEmitter.nDims)

              Code(
                storeElementIndex := x.pType.linearizeIndicesRowMajor(idxVars, reshapedShapeArray, elemMB),
                newIdxVarsSetup,
                childEmitter.outputElement(elemMB, newIdxVars)
              )
            }
          }

        case x@NDArrayConcat(nds, axis) =>
          val inputType = coerce[PArray](nds.pType)
          val inputNDType = coerce[PNDArray](inputType.elementType)

          val ndType = coerce[PNDArray](x.pType)
          val codeNDs = emit(nds)

          val inputArray = mb.genFieldThisRef[Long]()
          val n = mb.genFieldThisRef[Int]()
          val i = mb.genFieldThisRef[Int]()

          val loadAndValidateArray = Code(
            inputArray := codeNDs.value[Long],
            n := inputType.loadLength(inputArray),
            (n < 1).orEmpty(Code._fatal[Unit]("NDArrayConcat: can't concatenate 0 NDArrays")))

          val (missingSetup: Code[Unit @unchecked], missing: Code[Boolean @unchecked], setupShape: Code[Unit @unchecked]) = (inputType.required, inputNDType.required) match {
            case (true, true) => (Code._empty, false: Code[Boolean], Code(
              codeNDs.setup,
              codeNDs.m.orEmpty(Code._fatal[Unit]("NDArrayConcat: required NDArray can't be missing")),
              loadAndValidateArray))
            case (false, true) => (codeNDs.setup, codeNDs.m, loadAndValidateArray)
            case _ =>
              val m = mb.genFieldThisRef[Boolean]()
              val setup = Code(
                codeNDs.setup,
                m := codeNDs.m,
                (!m).orEmpty(
                  Code(
                    loadAndValidateArray,
                    i := 0,
                    Code.whileLoop(i < n,
                      m := m | inputType.isElementMissing(inputArray, i),
                      i := i + 1))
                ))
              (setup, m.load(), Code._empty)
          }

          val sb = SetupBuilder(mb, setupShape)
          val outputShape = sb.map(0 until ndType.nDims) { (sb, idx) =>
            val localDim = mb.genFieldThisRef[Long]()

            sb += Code(
              localDim := inputNDType.dimensionLength(inputType.loadElement(inputArray, 0), idx),
              i := 1,
              Code.whileLoop(i < n,
                {
                  if (idx == axis)
                    localDim := localDim + inputNDType.dimensionLength(inputType.loadElement(inputArray, i), idx)
                  else
                    inputNDType.dimensionLength(inputType.loadElement(inputArray, i), idx).cne(localDim)
                      .orEmpty(Code._fatal[Unit](
                        const(s"NDArrayConcat: mismatched dimensions of input NDArrays along axis $i: expected ")
                          .concat(localDim.toS).concat(", got ")
                          .concat(inputNDType.dimensionLength(inputType.loadElement(inputArray, i), idx).toS)))
                },
                i := i + 1))

            localDim
          }

          val setupShape2 = sb.result()

          new NDArrayEmitter[C](x.typ.nDims,
            outputShape,
            ndType.shape.pType,
            ndType.elementType,
            setupShape2,
            missingSetup,
            missing) {
            override def outputElement(elemMB: EmitMethodBuilder[C], idxVars: IndexedSeq[Value[Long]]): Code[_] = {
              val concatAxisIdx = elemMB.newLocal[Long]()

              val setupTransformedIdx = Code(
                i := 0,
                concatAxisIdx := idxVars(axis),
                Code.whileLoop(concatAxisIdx >= inputNDType.dimensionLength(inputType.loadElement(inputArray, i), axis),
                  concatAxisIdx := concatAxisIdx - inputNDType.dimensionLength(inputType.loadElement(inputArray, i), axis),
                  i := i + 1),
                (i > n).orEmpty(Code._fatal[Unit](
                  const("NDArrayConcat: trying to access element greater than length of concatenation axis: ")
                    .concat(i.toS).concat(" > ").concat(n.toS))))

              val transformedIdxs = Array.tabulate(x.typ.nDims) { idx =>
                if (idx == axis) concatAxisIdx else idxVars(idx)
              }.toFastIndexedSeq
              Code(
                setupTransformedIdx,
                inputNDType.loadElementToIRIntermediate(transformedIdxs, new Value[Long] {
                  def get: Code[Long] = inputType.loadElement(inputArray, i)
                }, elemMB))
            }
          }


        case x@NDArraySlice(child, slicesIR) =>
          val childEmitter = deforest(child)

          val slicest = emit(slicesIR)
          val slicesValueAddress = mb.genFieldThisRef[Long]("ndarr_slicev")
          val slicesm = mb.genFieldThisRef[Boolean]("ndarr_slicem")
          val slices = new CodePTuple(coerce[PTuple](slicesIR.pType), slicesValueAddress)

          val slicers = slices.withTypes.collect {
            case (t: PTuple, slice) => new CodePTuple(t, coerce[Long](slice))
          }

          val missingSliceElements = slicers.map(_.missingnessPattern.reduce(_ || _)).fold(false: Code[Boolean])(_ || _)
          val anyMissingness = missingSliceElements || slices.missingnessPattern.fold(false: Code[Boolean])(_ || _)

          val codeSlices = slicers.map(_.values[Long, Long, Long])

          val sb = SetupBuilder(mb, childEmitter.setupShape)
          val outputShape = codeSlices.zipWithIndex.map { case ((start, stop, step), i) =>
            sb.memoizeField(
              (step >= 0L && start <= stop).mux(
                const(1L) + ((stop - start) - 1L) / step,
                (step < 0L && start >= stop).mux(
                  (((stop - start) + 1L) / step) + 1L,
                  0L)),
              s"nda_slice_shape$i")
          }

          val setupShape = sb.result()

          val setupMissing = Code(childEmitter.setupMissing,
            slicesm := slicest.m,
            slicesValueAddress := slicesm.mux(0L, slicest.value[Long]),
            slicest.setup)

          val missing = childEmitter.missing || anyMissingness

          new NDArrayEmitter[C](x.pType.nDims, outputShape, x.pType.shape.pType, x.pType.elementType, setupShape, setupMissing, missing) {
            override def outputElement(elemMB: EmitMethodBuilder[C], idxVars: IndexedSeq[Value[Long]]): Code[_] = {
              val oldIdxVarsIter = idxVars.iterator

              val sliceIdxVars2: IndexedSeq[Value[Long]] = slices.withTypes.map {
                case (_: PInt64, indexer) =>
                  coerce[Long](indexer)
                case (t: PTuple, slicer) =>
                  val (start, _, step) = new CodePTuple(t, coerce[Long](slicer)).values[Long, Long, Long]
                  new Value[Long] {
                    def get: Code[Long] = start + oldIdxVarsIter.next() * step
                  }
              }

              childEmitter.outputElement(elemMB, sliceIdxVars2)
            }
          }

        case x@NDArrayFilter(child, filters) =>
          val childEmitter = deforest(child)

          val sb = SetupBuilder(mb, childEmitter.setupShape)

          val (vars, outputShape) = filters.zipWithIndex.map { case (f, i) =>
            val codeF = emit(f)
            val m = mb.genFieldThisRef[Boolean](s"m_filter$i")
            val v = mb.genFieldThisRef[Long](s"v_filter$i")

            val shapeVar = sb.memoizeField(Code(
                codeF.setup,
                m := codeF.m,
                m.mux(
                  Code(v := 0L, childEmitter.outputShape(i)),
                  Code(v := codeF.value[Long], coerce[PArray](f.pType).loadLength(v).toL))),
              s"nda_filter_shape$i")

            ((m, v), shapeVar)
          }.unzip

          val setupShape = sb.result()

          new NDArrayEmitter[C](x.pType.nDims, outputShape, x.pType.shape.pType, x.pType.elementType, setupShape, childEmitter.setupMissing, childEmitter.missing) {
            override def outputElement(elemMB: EmitMethodBuilder[C], idxVars: IndexedSeq[Value[Long]]): Code[_] = {
              val newIdxVars: IndexedSeq[Settable[Long]] = Array.tabulate(x.pType.nDims) { _ => mb.genFieldThisRef[Long]() }

              Code(
                Code(
                  Array.tabulate(x.pType.nDims) { i =>
                    val (m, v) = vars(i)
                    val typ = coerce[PArray](filters(i).pType)
                    newIdxVars(i) := m.mux(
                      idxVars(i),
                      typ.isElementMissing(v, idxVars(i).toI).mux(
                        Code._fatal[Long](s"NDArrayFilter: can't filter on missing index (axis=$i)"),
                        Region.loadLong(typ.loadElement(v.load(), idxVars(i).toI))))
                  }),
                childEmitter.outputElement(elemMB, newIdxVars))
            }
          }

        case _ =>
          val ndt = emit(x)
          val ndAddress = mb.genFieldThisRef[Long]()
          val setup = (ndAddress := ndt.value[Long])
          val xP = x.pType.asInstanceOf[PNDArray]

          val shapeAddress = new Value[Long] {
            def get: Code[Long] = xP.shape.load(ndAddress)
          }
          val shapeTuple = new CodePTuple(xP.shape.pType, shapeAddress)

          val shapeArray = (0 until xP.shape.pType.nFields).map(i => shapeTuple.apply[Long](i))

          new NDArrayEmitter[C](nDims, shapeArray,
            xP.shape.pType, xP.elementType, setup, ndt.setup, ndt.m) {
            override def outputElement(elemMB: EmitMethodBuilder[C], idxVars: IndexedSeq[Value[Long]]): Code[_] =
              xP.loadElementToIRIntermediate(idxVars, ndAddress, elemMB)
          }
      }
    }

    deforest(x0).emit(mb, coerce[PNDArray](x0.pType), region.code)
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

    setup = Code(setup,
      (lK cne rK).orEmpty(
        Code._fatal[Unit](const("Matrix dimensions incompatible: ").concat(lK.toS).concat(" ").concat(rK.toS))))

    cb.append(setup)
    shape
  }
}

abstract class NDArrayEmitter[C](
   val nDims: Int,
   val outputShape: IndexedSeq[Value[Long]],
   val outputShapePType: PTuple,
   val outputElementPType: PType,
   val setupShape: Code[Unit],
   val setupMissing: Code[Unit] = Code._empty,
   val missing: Code[Boolean] = false) {

  def outputElement(elemMB: EmitMethodBuilder[C], idxVars: IndexedSeq[Value[Long]]): Code[_]

  def emit(mb: EmitMethodBuilder[C], targetType: PNDArray, region: Value[Region]): EmitCode = {
    val outputShapeVariables = (0 until nDims).map(_ => mb.genFieldThisRef[Long]())

    val dataSrvb = new StagedRegionValueBuilder(mb, targetType.data.pType, region)

    val dataAddress: Code[Long] =
      Code(
        dataSrvb.start(targetType.numElements(outputShapeVariables, mb).toI),
        emitLoops(mb, outputShapeVariables, dataSrvb),
        dataSrvb.end())

    def shapeBuilder(srvb: StagedRegionValueBuilder): Code[Unit] = {
      coerce[Unit](Code(
        srvb.start(),
        Code.foreach(outputShapeVariables){ shapeElement =>
          Code(
            srvb.addLong(shapeElement),
            srvb.advance()
          )
        }
      ))
    }

    val m = mb.genFieldThisRef[Boolean]()

    val fullSetup = Code(
      setupMissing,
      m := missing,
      m.mux(
        Code._empty,
        Code(
          setupShape,
          Code.foreach(0 until nDims)(index => outputShapeVariables(index) := outputShape(index)))))

    val ptr = targetType.construct(
      shapeBuilder,
      targetType.makeColumnMajorStridesBuilder(outputShapeVariables, mb),
      dataAddress,
      mb,
      region)

    EmitCode(fullSetup, m, ptr)
  }

  private def emitLoops(mb: EmitMethodBuilder[C], outputShapeVariables: IndexedSeq[Value[Long]], srvb: StagedRegionValueBuilder): Code[Unit] = {
    val idxVars = Array.tabulate(nDims) { _ => mb.genFieldThisRef[Long]() }.toFastIndexedSeq
    val storeElement = mb.newLocal("nda_elem_out")(typeToTypeInfo(outputElementPType))

    val body =
      Code(
        storeElement.storeAny(outputElement(mb, idxVars)),
        srvb.addIRIntermediate(outputElementPType)(storeElement),
        srvb.advance()
      )

    val columnMajorLoops = idxVars.zipWithIndex.foldLeft(body) { case (innerLoops, (dimVar, dimIdx)) =>
      Code(
        dimVar := 0L,
        Code.whileLoop(dimVar < outputShapeVariables(dimIdx),
          innerLoops,
          dimVar := dimVar + 1L
        )
      )
    }

    columnMajorLoops
  }
}

abstract class NDArrayEmitter2(val outputShape: IEmitCodeGen[IndexedSeq[Value[Long]]])
{
  val nDims = outputShape.value.length

  def outputElement(cb: EmitCodeBuilder, idxVars: IndexedSeq[Value[Long]]): Code[_]

  def emit(cb: EmitCodeBuilder, targetType: PNDArray, region: Value[Region]): IEmitCode = {
    outputShape.map(cb){ shapeArray: IndexedSeq[Value[Long]] =>
      val dataSrvb = new StagedRegionValueBuilder(cb.emb, targetType.data.pType, region)
      cb.append(dataSrvb.start(targetType.numElements(shapeArray, cb.emb).toI))
      emitLoops(cb, shapeArray, targetType.elementType, dataSrvb)
      val dataAddress: Code[Long] = dataSrvb.end()

      def shapeBuilder(srvb: StagedRegionValueBuilder): Code[Unit] = {
        coerce[Unit](Code(
          srvb.start(),
          Code.foreach(shapeArray) { shapeElement =>
            Code(
              srvb.addLong(shapeElement),
              srvb.advance()
            )
          }
        ))
      }

      val ptr = targetType.construct(
        shapeBuilder,
        targetType.makeColumnMajorStridesBuilder(shapeArray, cb.emb),
        dataAddress,
        cb.emb,
        region)

      ptr
    }
  }

  private def emitLoops(cb: EmitCodeBuilder, outputShapeVariables: IndexedSeq[Value[Long]], outputElementPType: PType, srvb: StagedRegionValueBuilder): Unit = {
    val mb = cb.emb
    val idxVars = Array.tabulate(nDims) { i => cb.newField[Long](s"ndarray_emitter_idxVar_$i") }.toFastIndexedSeq
    val storeElement = mb.newLocal("nda_elem_out")(typeToTypeInfo(outputElementPType))

    val body =
      Code(
        storeElement.storeAny(outputElement(cb, idxVars)),
        srvb.addIRIntermediate(outputElementPType)(storeElement),
        srvb.advance()
      )

    val columnMajorLoops = idxVars.zipWithIndex.foldLeft(body) { case (innerLoops, (dimVar, dimIdx)) =>
      Code(
        dimVar := 0L,
        Code.whileLoop(dimVar < outputShapeVariables(dimIdx),
          innerLoops,
          dimVar := dimVar + 1L
        )
      )
    }

    cb.append(columnMajorLoops)
  }
}

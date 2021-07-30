package is.hail.expr.ir

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend.BackendContext
import is.hail.expr.ir.agg.{AggStateSig, ArrayAggStateSig, GroupedStateSig}
import is.hail.expr.ir.analyses.{ComputeMethodSplits, ControlFlowPreventsSplit, ParentPointers}
import is.hail.expr.ir.lowering.TableStageDependency
import is.hail.expr.ir.ndarrays.EmitNDArray
import is.hail.expr.ir.streams.{EmitStream, StreamProducer, StreamUtils}
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer, TypedCodecSpec}
import is.hail.linalg.{BLAS, LAPACK, LinalgCodeUtils}
import is.hail.services.shuffler._
import is.hail.types.physical._
import is.hail.types.physical.stypes._
import is.hail.types.physical.stypes.concrete._
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives._
import is.hail.types.virtual._
import is.hail.types.{TypeWithRequiredness, VirtualTypeWithReq}
import is.hail.utils._

import java.io._
import scala.collection.mutable
import scala.language.{existentials, postfixOps}

// class for holding all information computed ahead-of-time that we need in the emitter
object EmitContext {
  def analyze(ctx: ExecuteContext, ir: IR): EmitContext = {
    ctx.timer.time("EmitContext.analyze") {
      val usesAndDefs = ComputeUsesAndDefs(ir, errorIfFreeVariables = false)
      val requiredness = Requiredness.apply(ir, usesAndDefs, null, Env.empty) // Value IR inference doesn't need context
      val inLoopCriticalPath = ControlFlowPreventsSplit(ir, ParentPointers(ir), usesAndDefs)
      val methodSplits = ComputeMethodSplits(ir,inLoopCriticalPath)
      new EmitContext(ctx, requiredness, usesAndDefs, methodSplits, inLoopCriticalPath, Memo.empty[Unit])
    }
  }
}

class EmitContext(
  val executeContext: ExecuteContext,
  val req: RequirednessAnalysis,
  val usesAndDefs: UsesAndDefs,
  val methodSplits: Memo[Unit],
  val inLoopCriticalPath: Memo[Unit],
  val tryingToSplit: Memo[Unit]
)

case class EmitEnv(bindings: Env[EmitValue], inputValues: IndexedSeq[Value[Region] => EmitValue]) {
  def bind(name: String, v: EmitValue): EmitEnv = copy(bindings = bindings.bind(name, v))

  def bind(newBindings: (String, EmitValue)*): EmitEnv = copy(bindings = bindings.bindIterable(newBindings))
}

object Emit {
  def apply[C](ctx: EmitContext, ir: IR, fb: EmitFunctionBuilder[C], rti: TypeInfo[_], nParams: Int, aggs: Option[Array[AggStateSig]] = None): Option[SingleCodeType] = {
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
        val env = EmitEnv(Env.empty, (0 until nParams).map(i => mb.storeEmitParam(i + 2, cb))) // this, region, ...
        emitter.emitVoid(cb, ir, region, env, container, None)
      }
      None
    } else {
      var sct: SingleCodeType = null
      fb.emitWithBuilder { cb =>

        val env = EmitEnv(Env.empty, (0 until nParams).map(i => mb.storeEmitParam(i + 2, cb))) // this, region, ...
        val sc = emitter.emitI(ir, cb, region, env, container, None).handle(cb, {
          cb._throw[RuntimeException](
            Code.newInstance[RuntimeException, String]("cannot return empty"))
        })

        val scp = SingleCodeSCode.fromSCode(cb, sc, region)
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

    val cleanup = { cb: EmitCodeBuilder =>
      aggState.store(cb)
      cb += region.load().invalidate()
      cb.assign(region, Code._null[Region])
    }

    (AggContainer(aggs, aggState, () => ()), (cb: EmitCodeBuilder) => cb += setup, cleanup)
  }

  def fromMethodBuilder[C](aggs: Array[AggStateSig], mb: EmitMethodBuilder[C], varPrefix: String): (AggContainer, EmitCodeBuilder => Unit, EmitCodeBuilder => Unit) =
    fromVars(aggs, mb, mb.genFieldThisRef[Region](s"${ varPrefix }_top_region"), mb.genFieldThisRef[Long](s"${ varPrefix }_off"))

  def fromBuilder[C](cb: EmitCodeBuilder, aggs: Array[AggStateSig], varPrefix: String): AggContainer = {
    val off = cb.newField[Long](s"${ varPrefix }_off")
    val region = cb.newField[Region](s"${ varPrefix }_top_region", Region.stagedCreate(Region.REGULAR, cb.emb.ecb.pool()))
    val states = agg.StateTuple(aggs.map(a => agg.AggStateSig.getState(a, cb.emb.ecb)))
    val aggState = new agg.TupleAggregatorState(cb.emb.ecb, states, region, off)
    cb += region.load().setNumParents(aggs.length)
    cb.assign(off, region.load().allocate(aggState.storageType.alignment, aggState.storageType.byteSize))
    states.createStates(cb)

    AggContainer(aggs, aggState, { () =>
      aggState.store(cb)
      cb += region.load().invalidate()
      cb.assign(region, Code._null[Region])
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
  def emitType: EmitType

  def st: SType = emitType.st

  def load: EmitCode

  def get(cb: EmitCodeBuilder): SCode
}

class EmitUnrealizableValue(private val ec: EmitCode) extends EmitValue {
  val emitType: EmitType = ec.emitType

  assert(st.isInstanceOf[SStream])
  private[this] var used: Boolean = false

  def load: EmitCode = {
    assert(!used)
    used = true
    ec
  }

  override def get(cb: EmitCodeBuilder): SCode = throw new UnsupportedOperationException(s"Can't make PValue for unrealizable type ${ ec.st }")
}

/**
  * Notes on IEmitCode;
  *  1. It is the responsibility of the producers of IEmitCode to emit the relevant
  *     jumps for the Lmissing and Lpresent labels (cb.goto or similar)
  *  2. It is the responsibility of consumers to define these labels and to
  *     prevent the SCode from being used on any code path taken as a result of
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
    IEmitCodeGen(Lmissing, Lpresent, value, required)

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

  def multiMapEmitCodes(cb: EmitCodeBuilder, seq: IndexedSeq[EmitCode])(f: IndexedSeq[SCode] => SCode): IEmitCode = {
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

  implicit class IEmitCode(val iec: IEmitCodeGen[SCode]) extends AnyVal {
    def pc: SCode = iec.value

    def st: SType = pc.st

    def memoize(cb: EmitCodeBuilder, name: String): EmitValue =
      cb.memoize(iec, name)

    def memoizeField(cb: EmitCodeBuilder, name: String): EmitValue =
      cb.memoizeField(iec, name)
  }

}

case class IEmitCodeGen[+A](Lmissing: CodeLabel, Lpresent: CodeLabel, value: A, required: Boolean) {

  lazy val emitType: EmitType = {
    value match {
      case pc: SCode => EmitType(pc.st, required)
      case _ => throw new UnsupportedOperationException(s"emitType on $value")
    }
  }

  def setOptional: IEmitCodeGen[A] = copy(required = false)

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

  def get(cb: EmitCodeBuilder, errorMsg: Code[String]=s"expected non-missing", errorID: Code[Int] = const(ErrorIDs.NO_ERROR)): A =
    handle(cb, cb._fatalWithError(errorID, errorMsg))

  def consume(cb: EmitCodeBuilder, ifMissing: => Unit, ifPresent: (A) => Unit): Unit = {
    val Lafter = CodeLabel()
    cb.define(Lmissing)
    ifMissing
    if (cb.isOpenEnded) cb.goto(Lafter)
    cb.define(Lpresent)
    ifPresent(value)
    cb.define(Lafter)
  }

  def consumeSCode(cb: EmitCodeBuilder, ifMissing: => SCode, ifPresent: (A) => SCode): SCode = {
    val Lafter = CodeLabel()
    cb.define(Lmissing)
    val missingValue = ifMissing
    val st = missingValue.st
    val ret = cb.emb.newPLocal(st)
    cb.assign(ret, missingValue)
    cb.goto(Lafter)
    cb.define(Lpresent)
    val presentValue = ifPresent(value)
    assert(presentValue.st == st)
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
  def apply(setup: Code[Unit], m: Code[Boolean], pv: SCode): EmitCode = {
    Code.constBoolValue(m) match {
      case Some(false) =>
        val Lpresent = CodeLabel()
        new EmitCode(new CodeLabel(Code(setup, Lpresent.goto).start), IEmitCode(CodeLabel(), Lpresent, pv, required = true))
      case _ =>
        val mCC = Code(setup, m).toCCode
        val iec = IEmitCode(new CodeLabel(mCC.Ltrue), new CodeLabel(mCC.Lfalse), pv, required = false)
        val result = new EmitCode(new CodeLabel(mCC.entry), iec)
        result
    }
  }

  def unapply(ec: EmitCode): Option[(Code[Boolean], SCode)] =
    Some((ec.m, ec.pv))

  def apply(setup: Code[Unit], ec: EmitCode): EmitCode = {
    val Lstart = CodeLabel()
    Code(Lstart, setup, ec.start.goto)
    new EmitCode(Lstart, ec.iec)
  }

  def present(mb: EmitMethodBuilder[_], pc: SCode): EmitCode = EmitCode.fromI(mb)(cb => IEmitCode.present(cb, pc))

  def missing(mb: EmitMethodBuilder[_], pt: SType): EmitCode = EmitCode.fromI(mb)(cb => IEmitCode.missing(cb, pt.defaultValue))

  def fromI(mb: EmitMethodBuilder[_])(f: (EmitCodeBuilder) => IEmitCode): EmitCode = {
    val cb = EmitCodeBuilder(mb)
    val iec = f(cb)
    val setup = cb.result()
    new EmitCode(new CodeLabel(setup.start), iec)
  }
}

class EmitCode(private val start: CodeLabel, private val iec: IEmitCode) {

  def required: Boolean = iec.required

  def setOptional: EmitCode = new EmitCode(start, iec.setOptional)

  lazy val emitType: EmitType = iec.emitType

  def emitParamType: SCodeEmitParamType = emitType.paramType

  def st: SType = iec.value.st

  def pv: SCode = iec.value

  val m: Code[Boolean] = new CCode(start.L, iec.Lmissing.L, iec.Lpresent.L)

  def toI(cb: EmitCodeBuilder): IEmitCode = {
    cb.goto(start)
    iec
  }

  def castTo(mb: EmitMethodBuilder[_], region: Value[Region], destType: SType, deepCopy: Boolean = false): EmitCode = {
    EmitCode.fromI(mb)(cb => toI(cb).map(cb)(_.castTo(cb, region, destType)))
  }

  def makeCodeTuple(cb: EmitCodeBuilder): IndexedSeq[Code[_]] = {
    val ct = if (required) {
      toI(cb).get(cb).makeCodeTuple(cb)
    } else {
      val es = cb.emb.newEmitLocal("ec_makecodetuple", emitType)
      cb.assign(es, toI(cb))
      es.pv.makeCodeTuple(cb) :+ es.m
    }

    assert(ct.zip(emitParamType.codeTupleTypes).forall { case (p, pt) => p.ti == pt.ti},
      s"ctt mismatch: $emitType\n  param: ${ct.map(_.ti)}\n  types: ${emitParamType.codeTupleTypes}")
    ct
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

  def asVoid(): Code[Unit] = {
    require(pv.st == SVoid)
    Code.toUnit(m)
  }
}

object EmitSettable {
  def present(vs: SSettable): EmitSettable = new EmitSettable(None, vs)
}

class EmitSettable(
  missing: Option[Settable[Boolean]], // required if None
  vs: SSettable) extends EmitValue {

  lazy val required: Boolean = missing.isEmpty

  lazy val emitType: EmitType = EmitType(vs.st, required)

  def settableTuple(): IndexedSeq[Settable[_]] = {
    missing match {
      case Some(m) => vs.settableTuple() :+ m
      case None => vs.settableTuple()
    }
  }

  def m: Code[Boolean] = missing.map(_.load()).getOrElse(const(false))

  def load: EmitCode = {
    val ec = EmitCode(Code._empty,
      if (required) const(false) else missing.get.load(),
      vs.get)
    assert(ec.required == required)
    ec
  }

  def store(cb: EmitCodeBuilder, ec: EmitCode): Unit = {
    store(cb, ec.toI(cb))
  }

  def store(cb: EmitCodeBuilder, iec: IEmitCode): Unit =
    if (required)
      cb.assign(vs, iec.get(cb, s"Required EmitSettable cannot be missing ${ st }"))
    else
      iec.consume(cb, {
        cb.assign(missing.get, true)
      }, { value =>
        cb.assign(missing.get, false)
        cb.assign(vs, value)
      })

  override def get(cb: EmitCodeBuilder): SCode = {
    if (required) {
      vs
    } else {
      cb.ifx(missing.get, cb._fatal(s"Can't convert missing ${ st } to PValue"))
      vs
    }
  }
}

class RichIndexedSeqEmitSettable(is: IndexedSeq[EmitSettable]) {
  def load(): IndexedSeq[EmitCode] = is.map(_.load)
}

object LoopRef {
  def apply(cb: EmitCodeBuilder, L: CodeLabel, args: IndexedSeq[(String, EmitType)], pool: Value[RegionPool], resultType: EmitType): LoopRef = {
    val (loopArgs, tmpLoopArgs) = args.zipWithIndex.map { case ((name, et), i) =>
      (cb.emb.newEmitField(s"$name$i", et), cb.emb.newEmitField(s"tmp$name$i", et))
    }.unzip

    val r1: Settable[Region] = cb.newLocal[Region]("loop_ref_r1")
    cb.assign(r1, Region.stagedCreate(Region.REGULAR, pool))

    val r2: Settable[Region] = cb.newLocal[Region]("loop_ref_r2")
    cb.assign(r2, Region.stagedCreate(Region.REGULAR, pool))

    new LoopRef(L, args.map(_._2), loopArgs, tmpLoopArgs, r1, r2, resultType)
  }
}

class LoopRef(
  val L: CodeLabel,
  val loopTypes: IndexedSeq[EmitType],
  val loopArgs: IndexedSeq[EmitSettable],
  val tmpLoopArgs: IndexedSeq[EmitSettable],
  val r1: Settable[Region],
  val r2: Settable[Region],
  val resultType: EmitType)

abstract class EstimableEmitter[C] {
  def emit(mb: EmitMethodBuilder[C]): Code[Unit]

  def estimatedSize: Int
}

class Emit[C](
  val ctx: EmitContext,
  val cb: EmitClassBuilder[C]) {
  emitSelf =>

  val methods: mutable.Map[(String, Seq[Type], Seq[SType], SType), EmitMethodBuilder[C]] = mutable.Map()

  def emitVoidInSeparateMethod(context: String, cb: EmitCodeBuilder, ir: IR, region: Value[Region], env: EmitEnv, container: Option[AggContainer], loopEnv: Option[Env[LoopRef]]): Unit = {
    assert(!ctx.inLoopCriticalPath.contains(ir))
    val mb = cb.emb.genEmitMethod(context, FastIndexedSeq[ParamType](), UnitInfo)
    val r = cb.newField[Region]("emitVoidSeparate_region", region)
    mb.voidWithBuilder { cb =>
      ctx.tryingToSplit.bind(ir, ())
      emitVoid(cb, ir, r, env, container, loopEnv)
    }
    cb.invokeVoid(mb)
  }

  def emitSplitMethod(context: String, cb: EmitCodeBuilder, ir: IR, region: Value[Region], env: EmitEnv, container: Option[AggContainer], loopEnv: Option[Env[LoopRef]]): (EmitSettable, EmitMethodBuilder[_]) = {
    val mb = cb.emb.genEmitMethod(context, FastIndexedSeq[ParamType](), UnitInfo)
    val r = cb.newField[Region]("emitInSeparate_region", region)

    var ev: EmitSettable = null
    mb.voidWithBuilder { cb =>
      ctx.tryingToSplit.bind(ir, ())
      val result = emitI(ir, cb, r, env, container, loopEnv)

      ev = cb.emb.ecb.newEmitField(s"${context}_result", result.emitType)
      cb.assign(ev, result)
    }
    (ev, mb)
  }

  def emitInSeparateMethod(context: String, cb: EmitCodeBuilder, ir: IR, region: Value[Region], env: EmitEnv, container: Option[AggContainer], loopEnv: Option[Env[LoopRef]]): IEmitCode = {
    if (ir.typ == TVoid) {
      emitVoidInSeparateMethod(context, cb, ir, region, env, container, loopEnv)
      return IEmitCode.present(cb, SVoidCode)
    }

    assert(!ctx.inLoopCriticalPath.contains(ir))
    val (ev, mb) = emitSplitMethod(context, cb, ir, region, env, container, loopEnv)
    cb.invokeVoid(mb)
    ev.toI(cb)
  }

  private[ir] def emitVoid(cb: EmitCodeBuilder, ir: IR, region: Value[Region], env: EmitEnv, container: Option[AggContainer], loopEnv: Option[Env[LoopRef]]): Unit = {
    if (ctx.methodSplits.contains(ir) && !ctx.tryingToSplit.contains(ir)) {
      emitVoidInSeparateMethod(s"split_${ir.getClass.getSimpleName}", cb, ir, region, env, container, loopEnv)
      return
    }

    val mb: EmitMethodBuilder[C] = cb.emb.asInstanceOf[EmitMethodBuilder[C]]


    def emit(ir: IR, mb: EmitMethodBuilder[C] = mb, region: Value[Region] = region, env: EmitEnv = env, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): EmitCode =
      this.emit(ir, mb, region, env, container, loopEnv)

    def emitStream(ir: IR, outerRegion: Value[Region], mb: EmitMethodBuilder[C] = mb): EmitCode =
      EmitCode.fromI(mb)(cb => EmitStream.produce(this, ir, cb, outerRegion, env, container))

    def emitVoid(ir: IR, cb: EmitCodeBuilder = cb, region: Value[Region] = region, env: EmitEnv = env, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): Unit =
      this.emitVoid(cb, ir, region, env, container, loopEnv)

    def emitI(ir: IR, region: Value[Region] = region, env: EmitEnv = env, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): IEmitCode =
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

        cb.assign(ib, Code._null[InputBuffer])

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
        val aggStateOffset = mb.genFieldThisRef[Long](s"combOpValue_${ i }_state");

        val v = emitI(value)
        v.consume(cb,
          cb._fatal("cannot combOp a missing value"),
          { case serializedValue: SBinaryCode =>
            cb.assign(aggStateOffset, region.allocate(tempState.storageType.alignment, tempState.storageType.byteSize))
            tempState.createState(cb)
            tempState.newState(cb)
            tempState.deserializeFromBytes(cb, serializedValue)
            rvAgg.combOp(cb, sc.states(i), tempState)
          }
        )

      case InitFromSerializedValue(i, value, sig) =>
        val AggContainer(aggs, sc, _) = container.get
        assert(aggs(i) == sig)

        val v = emitI(value)
        v.consume(cb,
          cb._fatal("cannot initialize aggs from a missing value"),
          { case serializedValue: SBinaryCode =>
            sc.states(i).createState(cb)
            sc.newState(cb, i)
            sc.states(i).deserializeFromBytes(cb, serializedValue)
          }
        )
    }
  }

  private[ir] def emitI(ir: IR, cb: EmitCodeBuilder, env: EmitEnv, container: Option[AggContainer]): IEmitCode = {
    val region = cb.emb.getCodeParam[Region](1)
    emitI(ir, cb, region, env, container, None)
  }

  private[ir] def emitI(ir: IR, cb: EmitCodeBuilder, region: Value[Region], env: EmitEnv,
    container: Option[AggContainer], loopEnv: Option[Env[LoopRef]]
  ): IEmitCode = {
    if (ctx.methodSplits.contains(ir) && !ctx.tryingToSplit.contains(ir)) {
      return emitInSeparateMethod(s"split_${ir.getClass.getSimpleName}", cb, ir, region, env, container, loopEnv)
    }

    val mb: EmitMethodBuilder[C] = cb.emb.asInstanceOf[EmitMethodBuilder[C]]

    def emitI(ir: IR, region: Value[Region] = region, env: EmitEnv = env, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): IEmitCode =
      this.emitI(ir, cb, region, env, container, loopEnv)

    def emitInNewBuilder(cb: EmitCodeBuilder, ir: IR, region: Value[Region] = region, env: EmitEnv = env, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): IEmitCode =
      this.emitI(ir, cb, region, env, container, loopEnv)

    def emitStream(ir: IR, cb: EmitCodeBuilder, outerRegion: Value[Region]): IEmitCode =
      EmitStream.produce(this, ir, cb, outerRegion, env, container)

    def emitVoid(ir: IR, env: EmitEnv = env, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): Unit =
      this.emitVoid(cb, ir: IR, region, env, container, loopEnv)

    def emitFallback(ir: IR, env: EmitEnv = env, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): IEmitCode =
      this.emit(ir, mb, region, env, container, loopEnv, fallingBackFromEmitI = true).toI(cb)

    def emitDeforestedNDArrayI(ir: IR): IEmitCode = EmitNDArray(this, ir, cb, region, env, container, loopEnv)

    def emitNDArrayColumnMajorStrides(ir: IR): IEmitCode = {
      emitI(ir).map(cb) { case pNDCode: SNDArrayCode =>
        val pNDValue = pNDCode.memoize(cb, "ndarray_column_major_check")
        LinalgCodeUtils.checkColMajorAndCopyIfNeeded(pNDValue, cb, region)
      }
    }

    // Returns an IEmitCode along with a Boolean that is true if the returned value is column major. If false it's row
    // major instead.
    def emitNDArrayStandardStriding(ir: IR): IEmitCodeGen[(SNDArrayValue, Value[Boolean])] = {
      emitI(ir).map(cb) { case pNDCode: SNDArrayCode =>
        val pNDValue = pNDCode.memoize(cb, "ndarray_standard_striding_check")
        LinalgCodeUtils.checkStandardStriding(pNDValue, cb, region)
      }
    }

    def typeWithReqx(node: IR): VirtualTypeWithReq = VirtualTypeWithReq(node.typ, ctx.req.lookup(node).asInstanceOf[TypeWithRequiredness])

    def typeWithReq: VirtualTypeWithReq = typeWithReqx(ir)

    if (ir.typ == TVoid) {
      emitVoid(ir)
      return IEmitCode.present(cb, SVoidCode)
    }

    def presentPC(pc: SCode): IEmitCode = IEmitCode.present(cb, pc)

    val result: IEmitCode = (ir: @unchecked) match {
      case I32(x) =>
        presentPC(primitive(const(x)))
      case I64(x) =>
        presentPC(primitive(const(x)))
      case F32(x) =>
        presentPC(primitive(const(x)))
      case F64(x) =>
        presentPC(primitive(const(x)))
      case s@Str(x) =>
        presentPC(mb.addLiteral(x, typeWithReq))
      case x@UUID4(_) =>
        val pt = PCanonicalString()
        presentPC(pt.loadCheapSCode(cb, pt.
          allocateAndStoreString(mb, region, Code.invokeScalaObject0[String](
            Class.forName("is.hail.expr.ir.package$"), "uuid4"))))
      case x@Literal(t, v) =>
        presentPC(mb.addLiteral(v, typeWithReq))
      case x@EncodedLiteral(codec, value) =>
        presentPC(mb.addEncodedLiteral(x))
      case True() =>
        presentPC(primitive(const(true)))
      case False() =>
        presentPC(primitive(const(false)))
      case Consume(value) =>
        emitI(value).map(cb) { pc =>
          cb.memoizeField(pc, "consumed_field")
          // Ignore pc, just return a 1
          primitive(const(1L))
        }
      case Cast(v, typ) =>
        val iec = emitI(v)
        val cast = Casts.get(v.typ, typ)
        iec.map(cb)(pc => cast(cb, pc))
      case CastRename(v, _typ) =>
        emitI(v)
          .map(cb)(pc => pc.st.castRename(_typ).fromCodes(pc.makeCodeTuple(cb)))
      case NA(typ) =>
        IEmitCode.missing(cb, SUnreachable.fromVirtualType(typ).defaultValue)
      case IsNA(v) =>
        val m = emitI(v).consumeCode(cb, true, _ => false)
        presentPC(primitive(m))

      case Coalesce(values) =>

        val emittedValues = values.map(v => EmitCode.fromI(cb.emb)(cb => emitInNewBuilder(cb, v)))
        val unifiedType = SType.chooseCompatibleType(typeWithReq, emittedValues.map(_.st): _*)
        val coalescedValue = mb.newPLocal("coalesce_value", unifiedType)

        val Ldefined = CodeLabel()
        val Lmissing = CodeLabel()

        emittedValues.foreach { value =>
          value.toI(cb).consume(cb,
            {}, // fall through to next check
            { sc =>
              cb.assign(coalescedValue, sc.castTo(cb, region, unifiedType))
              cb.goto(Ldefined)
            })
        }

        cb.goto(Lmissing)

        IEmitCode(Lmissing, Ldefined, coalescedValue.load(), emittedValues.exists(_.required))

      case If(cond, cnsq, altr) =>
        assert(cnsq.typ == altr.typ)

        emitI(cond).flatMap(cb) { condValue =>


          val codeCnsq = EmitCode.fromI(cb.emb)(cb => emitInNewBuilder(cb, cnsq))
          val codeAltr = EmitCode.fromI(cb.emb)(cb => emitInNewBuilder(cb, altr))
          val outType = SType.chooseCompatibleType(typeWithReq, codeCnsq.st, codeAltr.st)

          val Lmissing = CodeLabel()
          val Ldefined = CodeLabel()
          val out = mb.newPLocal(outType)
          cb.ifx(condValue.asBoolean.boolCode(cb), {
            codeCnsq.toI(cb).consume(cb,
              {
                cb.goto(Lmissing)
              }, { sc =>
                cb.assign(out, sc.castTo(cb, region, outType))
              })
          }, {
            codeAltr.toI(cb).consume(cb,
              {
                cb.goto(Lmissing)
              }, { sc =>
                cb.assign(out, sc.castTo(cb, region, outType))
              })
          })
          cb.goto(Ldefined)

          IEmitCode(Lmissing, Ldefined, out.load(), codeCnsq.required && codeAltr.required)
        }

      case x@MakeStruct(fields) =>
        presentPC(SStackStruct.constructFromArgs(cb, region, x.typ.asInstanceOf[TBaseStruct],
          fields.map { case (_, x) =>
            EmitCode.fromI(cb.emb)(cb => emitInNewBuilder(cb, x))
          }: _*
        ))

      case x@MakeTuple(fields) =>
        presentPC(SStackStruct.constructFromArgs(cb, region, x.typ.asInstanceOf[TBaseStruct],
          fields.map { case (_, x) =>
            EmitCode.fromI(cb.emb)(cb => emitInNewBuilder(cb, x))
          }: _*
        ))

      case x@SelectFields(oldStruct, fields) =>
        emitI(oldStruct)
          .map(cb) { case sc: SBaseStructCode => sc.subset(fields: _*) }

      case x@InsertFields(old, fields, _) =>
        if (fields.isEmpty)
          emitI(old)
        else {
          emitI(old).map(cb) { old =>
            old.asBaseStruct.insert(cb, region, x.typ,
              fields.map { case (name, x) => (name, EmitCode.fromI(cb.emb)(cb => emitInNewBuilder(cb, x))) }: _*)
          }
        }

      case ApplyBinaryPrimOp(op, l, r) =>
        emitI(l).flatMap(cb) { pcL =>
          emitI(r).map(cb)(pcR => BinaryOp.emit(cb, op, pcL, pcR))
        }
      case ApplyUnaryPrimOp(op, x) =>
        emitI(x).map(cb)(pc => UnaryOp.emit(cb, op, pc))
      case ApplyComparisonOp(op, l, r) =>
        if (op.strict) {
          emitI(l).flatMap(cb) { l =>
            emitI(r).map(cb) { r =>
              val f = op.codeOrdering(cb.emb.ecb, l.st, r.st)
              primitive(ir.typ, f(cb, EmitCode.present(cb.emb, l), EmitCode.present(cb.emb, r)))
            }
          }
        } else {
          val lc = emitI(l).memoize(cb, "l")
          val rc = emitI(r).memoize(cb, "r")
          val f = op.codeOrdering(cb.emb.ecb, lc.st, rc.st)
          presentPC(primitive(ir.typ, f(cb, lc, rc)))
        }

      case x@MakeArray(args, _) =>

        val emittedArgs = args.map(a => EmitCode.fromI(mb)(cb => emitInNewBuilder(cb, a)))
        val pType = typeWithReq.canonicalPType.asInstanceOf[PCanonicalArray]
        val (pushElement, finish) = pType.constructFromFunctions(cb, region, args.size, deepCopy = false)
        for (arg <- emittedArgs) {
          pushElement(cb, arg.toI(cb))
        }
        presentPC(finish(cb))

      case ArrayZeros(length) =>
        emitI(length).map(cb) { case n: SInt32Code =>
          val outputPType = PCanonicalArray(PInt32Required)
          val elementSize = outputPType.elementByteSize
          val numElements = cb.newLocal[Int]("n_elements", n.intCode(cb))
          val arrayAddress = cb.newLocal[Long]("array_addr", outputPType.allocate(region, numElements))
          cb += outputPType.stagedInitialize(arrayAddress, numElements)
          cb += Region.setMemory(outputPType.firstElementOffset(arrayAddress), numElements.toL * elementSize, 0.toByte)
          outputPType.loadCheapSCode(cb, arrayAddress)
        }

      case x@ArrayRef(a, i, errorID) =>
        def boundsCheck(cb: EmitCodeBuilder, index: Value[Int], len: Value[Int]): Unit = {
            val bcMb = mb.getOrGenEmitMethod("arrayref_bounds_check", "arrayref_bounds_check",
              IndexedSeq[ParamType](IntInfo, IntInfo, IntInfo), UnitInfo)({ mb =>
              mb.voidWithBuilder { cb =>
                val index = mb.getCodeParam[Int](1)
                val len = mb.getCodeParam[Int](2)
                val errorID = mb.getCodeParam[Int](3)
                cb.ifx(index < 0 || index >= len, {
                  cb._fatalWithError(errorID, const("array index out of bounds: index=")
                    .concat(index.toS)
                    .concat(", length=")
                    .concat(len.toS))
                })

              }
            })
            cb.invokeVoid(bcMb, index, len, const(errorID))
        }

        emitI(a).flatMap(cb) { (ac) =>
          emitI(i).flatMap(cb) { (ic) =>
            val av = ac.asIndexable.memoize(cb, "aref_a")
            val iv = cb.newLocal("i", ic.asInt.intCode(cb))
            boundsCheck(cb, iv, av.loadLength())
            av.loadElement(cb, iv)
          }
        }

      case ArrayLen(a) =>
        emitI(a).map(cb) { (ac) =>
          primitive(ac.asIndexable.codeLoadLength())
        }

      case GetField(o, name) =>
        emitI(o).flatMap(cb) { oc =>
          oc.asBaseStruct.loadSingleField(cb, name)
        }

      case GetTupleElement(o, i) =>
        emitI(o).flatMap(cb) { oc =>
          oc.asBaseStruct.loadSingleField(cb, o.typ.asInstanceOf[TTuple].fieldIndex(i))
        }

      case x@LowerBoundOnOrderedCollection(orderedCollection, elem, onKey) =>
        emitI(orderedCollection).map(cb) { a =>
          val e = EmitCode.fromI(cb.emb)(cb => this.emitI(elem, cb, region, env, container, loopEnv))
          val bs = new BinarySearch[C](mb, a.st.asInstanceOf[SContainer], e.emitType, keyOnly = onKey)
          primitive(bs.getClosestIndex(cb, a, e))
        }

      case x@ArraySort(a, left, right, lessThan) =>
        emitStream(a, cb, region).map(cb) { case stream: SStreamCode =>
          val producer = stream.producer

          val sct = SingleCodeType.fromSType(producer.element.st)

          val vab = new StagedArrayBuilder(sct, producer.element.required, mb, 0)
          StreamUtils.writeToArrayBuilder(cb, stream.producer, vab, region)
          val sorter = new ArraySorter(EmitRegion(mb, region), vab)
          sorter.sort(cb, region, makeDependentSortingFunction(cb, sct, lessThan, env, emitSelf, Array(left, right)))
          sorter.toRegion(cb, x.typ)
        }

      case x@ToSet(a) =>
        emitStream(a, cb, region).map(cb) { case stream: SStreamCode =>
          val producer = stream.producer

          val sct = SingleCodeType.fromSType(producer.element.st)

          val vab = new StagedArrayBuilder(sct, producer.element.required, mb, 0)
          StreamUtils.writeToArrayBuilder(cb, stream.producer, vab, region)
          val sorter = new ArraySorter(EmitRegion(mb, region), vab)

          def lessThan(cb: EmitCodeBuilder, region: Value[Region], l: Code[_], r: Code[_]): Code[Boolean] = {
            cb.emb.ecb.getOrdering(sct.loadedSType, sct.loadedSType)
              .ltNonnull(cb, sct.loadToSCode(cb, region, l), sct.loadToSCode(cb, region, r))
          }

          sorter.sort(cb, region, lessThan)

          def skipNext(cb: EmitCodeBuilder, region: Value[Region], l: EmitCode, r: EmitCode): Code[Boolean] = {
            cb.newLocal[Boolean]("asdb", cb.emb.ecb.getOrdering(l.st, r.st)
              .equiv(cb, l, r, missingEqual = true))
          }

          sorter.distinctFromSorted(cb, region, skipNext)
          sorter.toRegion(cb, x.typ)
        }

      case x@ToDict(a) =>
        emitStream(a, cb, region).map(cb) { case stream: SStreamCode =>
          val producer = stream.producer

          val sct = SingleCodeType.fromSType(producer.element.st)

          val vab = new StagedArrayBuilder(sct, producer.element.required, mb, 0)
          StreamUtils.writeToArrayBuilder(cb, stream.producer, vab, region)
          val sorter = new ArraySorter(EmitRegion(mb, region), vab)

          def lessThan(cb: EmitCodeBuilder, region: Value[Region], l: Code[_], r: Code[_]): Code[Boolean] = {
            val lk = EmitCode.fromI(cb.emb)(cb => sct.loadToSCode(cb, region, l)
              .asBaseStruct.memoize(cb, "lt_l")
              .loadField(cb, 0))

            val rk = EmitCode.fromI(cb.emb)(cb => sct.loadToSCode(cb, region, r)
              .asBaseStruct.memoize(cb, "lt_r")
              .loadField(cb, 0))

            cb.emb.ecb.getOrdering(lk.st, rk.st)
              .lt(cb, lk, rk, missingEqual = true)
          }

          sorter.sort(cb, region, lessThan)
          sorter.pruneMissing(cb)

          def skipNext(cb: EmitCodeBuilder, region: Value[Region], l: EmitCode, r: EmitCode): Code[Boolean] = {

            val lk = EmitCode.fromI(cb.emb) { cb =>
              l.toI(cb).flatMap(cb) { x =>
                x.asBaseStruct.memoize(cb, "lt_l")
                  .loadField(cb, 0)
              }
            }

            val rk = EmitCode.fromI(cb.emb) { cb =>
              r.toI(cb).flatMap(cb) { x =>
                x.asBaseStruct.memoize(cb, "lt_r")
                  .loadField(cb, 0)
              }
            }

            cb.emb.ecb.getOrdering(lk.st, rk.st)
              .equiv(cb, lk, rk, missingEqual = true)
          }

          sorter.distinctFromSorted(cb, region, skipNext)
          sorter.toRegion(cb, x.typ)
        }

      case GroupByKey(collection) =>
        emitStream(collection, cb, region).map(cb) { case stream: SStreamCode =>

          val sct = SingleCodeType.fromSType(stream.producer.element.st)
          val sortedElts = new StagedArrayBuilder(sct, stream.producer.element.required, mb, 16)
          StreamUtils.writeToArrayBuilder(cb, stream.producer, sortedElts, region)
          val sorter = new ArraySorter(EmitRegion(mb, region), sortedElts)

          def lt(cb: EmitCodeBuilder, region: Value[Region], l: Code[_], r: Code[_]): Code[Boolean] = {
            val lk = EmitCode.fromI(cb.emb)(cb => sct.loadToSCode(cb, region, l)
              .asBaseStruct.memoize(cb, "lt_l")
              .loadField(cb, 0))
            val rk = EmitCode.fromI(cb.emb)(cb => sct.loadToSCode(cb, region, r)
              .asBaseStruct.memoize(cb, "lt_r")
              .loadField(cb, 0))
            cb.emb.ecb.getOrdering(lk.st, rk.st)
              .lt(cb, lk, rk, missingEqual = true)
          }

          sorter.sort(cb, region, lt)
          sorter.pruneMissing(cb)

          val groupSizes = new StagedArrayBuilder(Int32SingleCodeType, true, mb, 0)

          val eltIdx = mb.newLocal[Int]("groupByKey_eltIdx")
          val grpIdx = mb.newLocal[Int]("groupByKey_grpIdx")
          val withinGrpIdx = mb.newLocal[Int]("groupByKey_withinGrpIdx")
          val outerSize = mb.newLocal[Int]("groupByKey_outerSize")
          val groupSize = mb.newLocal[Int]("groupByKey_groupSize")


          cb += groupSizes.clear
          cb.assign(eltIdx, 0)
          cb.assign(groupSize, 0)

          def sameKeyAtIndices(cb: EmitCodeBuilder, region: Value[Region], idx1: Code[Int], idx2: Code[Int]): Code[Boolean] = {
            val lk = EmitCode.fromI(cb.emb) { cb =>
              sortedElts.loadFromIndex(cb, region, idx1).flatMap(cb) { x =>
                x.asBaseStruct.memoize(cb, "lt_l")
                  .loadField(cb, 0)
              }
            }

            val rk = EmitCode.fromI(cb.emb) { cb =>
              sortedElts.loadFromIndex(cb, region, idx2).flatMap(cb) { x =>
                x.asBaseStruct.memoize(cb, "lt_r")
                  .loadField(cb, 0)
              }
            }

            cb.emb.ecb.getOrdering(lk.st, rk.st)
              .equiv(cb, lk, rk, missingEqual = true)
          }

          cb.whileLoop(eltIdx < sortedElts.size, {
            val bottomOfLoop = CodeLabel()
            val newGroup = CodeLabel()

            cb.assign(groupSize, groupSize + 1)
            cb.ifx(eltIdx.ceq(sortedElts.size - 1), {
              cb.goto(newGroup)
            }, {
              cb.ifx(sameKeyAtIndices(cb, region, eltIdx, eltIdx + 1), {
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
          val loadedElementType = sct.loadedSType.asInstanceOf[SBaseStruct]
          val innerType = PCanonicalArray(loadedElementType.fieldEmitTypes(1).canonicalPType, true)
          val kt = loadedElementType.fieldEmitTypes(0).canonicalPType
          val groupType = PCanonicalStruct(true, ("key", kt), ("value", innerType))
          val dictType = PCanonicalDict(kt, innerType, false)
          val (addGroup, finishOuter) = dictType.arrayRep.constructFromFunctions(cb, region, outerSize, deepCopy = false)

          cb.assign(eltIdx, 0)
          cb.assign(grpIdx, 0)

          cb.whileLoop(grpIdx < outerSize, {
            cb.assign(groupSize, coerce[Int](groupSizes(grpIdx)))
            cb.assign(withinGrpIdx, 0)
            val firstStruct = sortedElts.loadFromIndex(cb, region, eltIdx).get(cb).asBaseStruct.memoize(cb, "GroupByKey_firstStruct")
            val key = EmitCode.fromI(mb) { cb => firstStruct.loadField(cb, 0) }
            val group = EmitCode.fromI(mb) { cb =>
              val (addElt, finishInner) = innerType
                .constructFromFunctions(cb, region, groupSize, deepCopy = false)
              cb.whileLoop(withinGrpIdx < groupSize, {
                val struct = sortedElts.loadFromIndex(cb, region, eltIdx).get(cb).asBaseStruct.memoize(cb, "GroupByKey_struct")
                addElt(cb, struct.loadField(cb, 1))
                cb.assign(eltIdx, eltIdx + 1)
                cb.assign(withinGrpIdx, withinGrpIdx + 1)
              })
              IEmitCode.present(cb, finishInner(cb))
            }
            val elt = groupType.constructFromFields(cb, region, FastIndexedSeq(key, group), deepCopy = false)
            addGroup(cb, IEmitCode.present(cb, elt))
            cb.assign(grpIdx, grpIdx + 1)
          })

          dictType.construct(finishOuter(cb))
        }

      case x@StreamLen(a) =>
        emitStream(a, cb, region).map(cb) { case stream: SStreamCode =>
          val producer = stream.producer
          producer.length match {
            case Some(compLen) =>
              producer.initialize(cb)
              val xLen = cb.newLocal[Int]("streamlen_x", compLen(cb))
              producer.close(cb)
              primitive(xLen)
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
              primitive(count)
          }
        }


      case x@MakeNDArray(dataIR, shapeIR, rowMajorIR, errorId) =>
        val nDims = x.typ.nDims

        emitI(rowMajorIR).flatMap(cb) { isRowMajorCode =>
          emitI(shapeIR).flatMap(cb) { case shapeTupleCode: SBaseStructCode =>
            dataIR.typ match {
              case _: TArray =>
                emitI(dataIR).map(cb) { case dataCode: SIndexableCode =>
                  val xP = PCanonicalNDArray(dataCode.st.elementType.canonicalPType().setRequired(true), nDims)
                  val shapeTupleValue = shapeTupleCode.memoize(cb, "make_ndarray_shape")
                  val memoData = dataCode.memoize(cb, "make_nd_array_memoized_data")

                  cb.ifx(memoData.hasMissingValues(cb), {
                    cb._throw(Code.newInstance[HailException, String, Int](
                      "Cannot construct an ndarray with missing values.", errorId
                    ))
                  })

                  (0 until nDims).foreach { index =>
                    cb.ifx(shapeTupleValue.isFieldMissing(index),
                      cb._fatalWithError(errorId, s"shape missing at index $index"))
                  }

                  val stridesSettables = (0 until nDims).map(i => cb.newLocal[Long](s"make_ndarray_stride_$i"))

                  val shapeValues = (0 until nDims).map { i =>
                    val shape = SingleCodeSCode.fromSCode(cb, shapeTupleValue.loadField(cb, i).get(cb), region)
                    cb.newLocalAny[Long](s"make_ndarray_shape_${ i }", shape.code)
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

                  xP.constructByCopyingArray(shapeValues, stridesSettables, memoData.asIndexable, cb, region)
                }
              case _: TStream =>
                EmitStream.produce(this, dataIR, cb, region, env, container)
                  .map(cb) {
                    case stream: SStreamCode => {
                      val xP = PCanonicalNDArray(stream.st.elementType.canonicalPType().setRequired(true), nDims)
                      val shapeTupleValue = shapeTupleCode.memoize(cb, "make_ndarray_shape")
                      (0 until nDims).foreach { index =>
                        cb.ifx(shapeTupleValue.isFieldMissing(index),
                          cb.append(Code._fatal[Unit](s"shape missing at index $index")))
                      }

                      val stridesSettables = (0 until nDims).map(i => cb.newLocal[Long](s"make_ndarray_stride_$i"))

                      val shapeValues = (0 until nDims).map { i =>
                        cb.newLocal[Long](s"make_ndarray_shape_${i}", shapeTupleValue.loadField(cb, i).get(cb).asLong.longCode(cb))
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

                      val (firstElementAddress, finisher) = xP.constructDataFunction(shapeValues, stridesSettables, cb, region)
                      StreamUtils.storeNDArrayElementsAtAddress(cb, stream.producer, region, firstElementAddress, errorId)
                      finisher(cb)
                    }
                  }
            }
          }
        }
      case NDArrayShape(ndIR) =>
        emitI(ndIR).map(cb) { case pc: SNDArrayCode => pc.shape(cb) }
      case x@NDArrayReindex(child, indexMap) =>
        val childEC = emitI(child)
        childEC.map(cb) { case sndCode: SNDArrayPointerCode =>
          val childPType = sndCode.st.pType
          val sndVal = sndCode.memoize(cb, "ndarray_reindex_child")
          val childShape = sndVal.shapes(cb)
          val childStrides = sndVal.strides(cb)

          val pndAddr = SingleCodeSCode.fromSCode(cb, sndVal, region)
          val dataPtr = sndVal.firstDataAddress(cb)

          val newShape = indexMap.map { childIndex =>
            if (childIndex < childPType.nDims) childShape(childIndex) else const(1L)
          }
          val newStrides = indexMap.map { childIndex =>
            if (childIndex < childPType.nDims) childStrides(childIndex) else const(0L)
          }

          val newPType = childPType.copy(nDims = indexMap.length)
          newPType.constructByCopyingDataPointer(
            newShape,
            newStrides,
            dataPtr,
            cb,
            region)
        }

      case NDArrayRef(nd, idxs, errorId) =>
        val ndt = emitI(nd)

        ndt.flatMap(cb) { case ndCode: SNDArrayCode =>
          val indexEmitCodes = idxs.map(idx => EmitCode.fromI(cb.emb)(emitInNewBuilder(_, idx)))
          IEmitCode.multiMapEmitCodes(cb, indexEmitCodes) { idxPCodes: IndexedSeq[SCode] =>
            val idxValues = idxPCodes.zipWithIndex.map { case (pc, idx) =>
              cb.newLocal(s"ref_idx_$idx", pc.asInt64.longCode(cb))
            }

            val ndValue = ndCode.memoize(cb, "reffed_ndarray")
            cb.append(ndValue.assertInBounds(idxValues, cb, errorId))

            ndValue.loadElement(idxValues, cb)
          }
        }

      case NDArrayMatMul(lChild, rChild, errorID) =>
        emitNDArrayStandardStriding(lChild).flatMap(cb) { case (leftPVal: SNDArrayValue, leftIsColumnMajor: Value[Boolean]) =>
          emitNDArrayStandardStriding(rChild).map(cb) { case (rightPVal: SNDArrayValue, rightIsColumnMajor: Value[Boolean]) =>
            val lSType = leftPVal.st
            val rSType = rightPVal.st

            val lShape = leftPVal.shapes(cb)
            val rShape = rightPVal.shapes(cb)

            val unifiedShape = NDArrayEmitter.matmulShape(cb, lShape, rShape, errorID)

            val leftBroadcastMask = if (lSType.nDims > 2) NDArrayEmitter.broadcastMask(lShape) else IndexedSeq[Value[Long]]()
            val rightBroadcastMask = if (rSType.nDims > 2) NDArrayEmitter.broadcastMask(rShape) else IndexedSeq[Value[Long]]()

            val outputPType = PCanonicalNDArray(lSType.elementType.canonicalPType().setRequired(true),
              TNDArray.matMulNDims(lSType.nDims, rSType.nDims))

            if ((lSType.elementType.virtualType == TFloat64 || lSType.elementType.virtualType == TFloat32) && lSType.nDims == 2 && rSType.nDims == 2) {
              val leftDataAddress = leftPVal.firstDataAddress(cb)
              val rightDataAddress = rightPVal.firstDataAddress(cb)

              val M = lShape(lSType.nDims - 2)
              val N = rShape(rSType.nDims - 1)
              val K = lShape(lSType.nDims - 1)

              val LDA = leftIsColumnMajor.mux(M, K)
              val LDB = rightIsColumnMajor.mux(K, N)
              val LDC = M

              val TRANSA: Code[String] = leftIsColumnMajor.mux("N", "T")
              val TRANSB: Code[String] = rightIsColumnMajor.mux("N", "T")

              val (answerFirstElementAddr, answerFinisher) = outputPType.constructDataFunction(
                IndexedSeq(M, N),
                outputPType.makeColumnMajorStrides(IndexedSeq(M, N), region, cb),
                cb,
                region)

              cb.ifx((M cne 0L) && (N cne 0L) && (K cne 0L), {
                cb.append(lSType.elementType.virtualType match {
                  case TFloat32 =>
                    Code.invokeScalaObject13[String, String, Int, Int, Int, Float, Long, Int, Long, Int, Float, Long, Int, Unit](BLAS.getClass, method = "sgemm",
                      TRANSA,
                      TRANSB,
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
                  case TFloat64 =>
                    Code.invokeScalaObject13[String, String, Int, Int, Int, Double, Long, Int, Long, Int, Double, Long, Int, Unit](BLAS.getClass, method = "dgemm",
                      TRANSA,
                      TRANSB,
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
            } else if (lSType.elementType.virtualType == TFloat64 && lSType.nDims == 2 && rSType.nDims == 1) {
              val leftDataAddress = leftPVal.firstDataAddress(cb)
              val rightDataAddress = rightPVal.firstDataAddress(cb)

              val numRows = lShape(lSType.nDims - 2)
              val numCols = lShape(lSType.nDims - 1)
              val M = cb.newLocal[Long]("dgemv_m", leftIsColumnMajor.mux(numRows, numCols))
              val N = cb.newLocal[Long]("dgemv_n", leftIsColumnMajor.mux(numCols, numRows))
              val outputSize = cb.newLocal[Long]("output_size", numRows)

              val alpha = 1.0
              val beta = 0.0

              val LDA = M
              val TRANS: Code[String] = leftIsColumnMajor.mux("N", "T")

              val (answerFirstElementAddr, answerFinisher) = outputPType.constructDataFunction(
                IndexedSeq(outputSize),
                outputPType.makeColumnMajorStrides(IndexedSeq(outputSize), region, cb),
                cb,
                region)
              
              cb.append(Code.invokeScalaObject11[String, Int, Int, Double, Long, Int, Long, Int, Double, Long, Int, Unit](BLAS.getClass, method="dgemv",
                TRANS,
                M.toI,
                N.toI,
                alpha,
                leftDataAddress,
                LDA.toI,
                rightDataAddress,
                1,
                beta,
                answerFirstElementAddr,
                1
              ))


              answerFinisher(cb)
            }  else {
              val numericElementType = coerce[PNumeric](lSType.elementType.canonicalPType())
              val eVti = typeToTypeInfo(numericElementType)

              val emitter = new NDArrayEmitter(unifiedShape, leftPVal.st.elementType) {
                override def outputElement(cb: EmitCodeBuilder, idxVars: IndexedSeq[Value[Long]]): SCode = {
                  val element = coerce[Any](cb.newField("matmul_element")(eVti))
                  val k = cb.newField[Long]("ndarray_matmul_k")

                  val (lIndices: IndexedSeq[Value[Long]], rIndices: IndexedSeq[Value[Long]]) = (lSType.nDims, rSType.nDims, idxVars) match {
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

                  def multiply(l: SCode, r: SCode): Code[_] = {
                    (l.st, r.st) match {
                      case (SInt32, SInt32) =>
                        l.asInt.intCode(cb) * r.asInt.intCode(cb)
                      case (SInt64, SInt64) =>
                        l.asLong.longCode(cb) * r.asLong.longCode(cb)
                      case (SFloat32, SFloat32) =>
                        l.asFloat.floatCode(cb) * r.asFloat.floatCode(cb)
                      case (SFloat64, SFloat64) =>
                        l.asDouble.doubleCode(cb) * r.asDouble.doubleCode(cb)
                    }
                  }

                  cb.assign(kLen, lShape(lSType.nDims - 1))
                  cb.assign(element, numericElementType.zero)
                  cb.forLoop(cb.assign(k, 0L), k < kLen, cb.assign(k, k + 1L), {
                    val lElem = leftPVal.loadElement(lIndices, cb)
                    val rElem = rightPVal.loadElement(rIndices, cb)
                    cb.assign(element, numericElementType.add(multiply(lElem, rElem), element))
                  })

                  primitive(outputPType.elementType.virtualType, element)
                }
              }
              emitter.emit(cb, outputPType, region)
            }
          }
        }
      case NDArrayInv(nd, errorID) =>
        // Based on https://github.com/numpy/numpy/blob/v1.19.0/numpy/linalg/linalg.py#L477-L547
        emitNDArrayColumnMajorStrides(nd).map(cb) { case pNDCode: SNDArrayCode =>
          val pndVal = pNDCode.memoize(cb, "ndarray_inverse_nd")
          val ndPT = pndVal.st.asInstanceOf[SNDArrayPointer].pType

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
            .orEmpty(Code._fatalWithID[Unit](const(s"LAPACK error ${ fun }. Error code = ").concat(info.toS), const(errorID)))

          cb.append((N cne M).orEmpty(Code._fatalWithID[Unit](const("Can only invert square matrix"), const(errorID))))

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
      case x@NDArraySVD(nd, full_matrices, computeUV, errorID) =>
        emitNDArrayColumnMajorStrides(nd).flatMap(cb) { case ndPCode: SNDArrayCode =>
          val ndPVal = ndPCode.memoize(cb, "nd_svd_value")

          val infoDGESDDResult = cb.newLocal[Int]("infoDGESDD")
          val infoDGESDDErrorTest = (extraErrorMsg: String) => (infoDGESDDResult cne 0)
            .orEmpty(Code._fatalWithID[Unit](const(s"LAPACK error DGESDD. $extraErrorMsg Error code = ").concat(infoDGESDDResult.toS), errorID))

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

          cb.assign(LWORKAddress, Code.invokeStatic1[Memory, Long, Long]("malloc", 8L))

          val retPTypeUncast = NDArraySVD.pTypes(computeUV, false)

          val (jobz, sPType, uData, uFinisher, vtData, vtFinisher) = if (computeUV) {
            val outputPType = retPTypeUncast.asInstanceOf[PTuple]
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
            val outputPType = retPTypeUncast.asInstanceOf[PCanonicalNDArray]

            def noOp(cb: EmitCodeBuilder): SNDArrayCode = {
              throw new IllegalStateException("Can't happen")
            }

            ("N", outputPType.asInstanceOf[PCanonicalNDArray], const(0L), noOp(_), const(0L), noOp(_))
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

            val outputPType = NDArraySVD.pTypes(true, false).asInstanceOf[PCanonicalTuple]
            outputPType.constructFromFields(cb, region, FastIndexedSeq(EmitCode.present(cb.emb, u), EmitCode.present(cb.emb, s), EmitCode.present(cb.emb, vt)), deepCopy = false)
          } else {
            s
          }
          IEmitCode(cb, false, resultPCode)

        }
      case x@NDArrayQR(nd, mode, errorID) =>
        // See here to understand different modes: https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.qr.html
        emitNDArrayColumnMajorStrides(nd).map(cb) { case pndCode: SNDArrayCode =>

          val resultPType = NDArrayQR.pType(mode, false)

          val pndValue = pndCode.memoize(cb, "ndarray_qr_nd")
          // This does a lot of byte level copying currently, so only trust
          // the PCanonicalNDArray representation.
          val pType = pndValue.st.asInstanceOf[SNDArrayPointer].pType

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

          val ndPT = pType
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
            .orEmpty(Code._fatalWithID[Unit](const(s"LAPACK error DGEQRF. $extraErrorMsg Error code = ").concat(infoDGEQRFResult.toS), errorID))

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

          val result: SCode = if (mode == "raw") {
            val resultType = resultPType.asInstanceOf[PCanonicalBaseStruct]
            val tau = tauFinisher(cb)

            resultType.constructFromFields(cb, region, FastIndexedSeq(
              EmitCode.present(cb.emb, hMemo),
              EmitCode.present(cb.emb, tau)
            ), deepCopy = false)

          } else {
            val (rPType, rRows, rCols) = if (mode == "r") {
              (resultPType.asInstanceOf[PCanonicalNDArray], K, N)
            } else if (mode == "complete") {
              (resultPType.asInstanceOf[PTuple].types(1).asInstanceOf[PCanonicalNDArray], M, N)
            } else if (mode == "reduced") {
              (resultPType.asInstanceOf[PTuple].types(1).asInstanceOf[PCanonicalNDArray], K, N)
            } else {
              throw new AssertionError(s"Unsupported QR mode $mode")
            }

            val rShapeArray = FastIndexedSeq[Value[Long]](rRows, rCols)

            val rStridesArray = rPType.makeColumnMajorStrides(rShapeArray, region, cb)

            val (rDataAddress, rFinisher) = rPType.constructDataFunction(rShapeArray, rStridesArray, cb, region)

            // This block assumes that `rDataAddress` and `aAddressDGEQRF` point to column major arrays.
            // TODO: Abstract this into ndarray ptype/SCode interface methods.
            val currRow = cb.newLocal[Long]("ndarray_qr_currRow")
            val currCol = cb.newLocal[Long]("ndarray_qr_currCol")

            val curWriteAddress = cb.newLocal[Long]("ndarray_qr_curr_write_addr", rDataAddress)

            // I think this just copies out the upper triangle into new ndarray in column major order
            cb.forLoop({
              cb.assign(currCol, 0L)
            }, currCol < rCols, {
              cb.assign(currCol, currCol + 1L)
            }, {
              cb.forLoop({
                cb.assign(currRow, 0L)
              }, currRow < rRows, {
                cb.assign(currRow, currRow + 1L)
              }, {
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
              val crPType = resultPType.asInstanceOf[PCanonicalTuple]

              val qPType = crPType.types(0).asInstanceOf[PCanonicalNDArray]
              val qShapeArray = if (mode == "complete") Array(M, M) else Array(M, K)
              val qStridesArray = qPType.makeColumnMajorStrides(qShapeArray, region, cb)

              val infoDORGQRResult = cb.newLocal[Int]("ndarray_qr_DORGQR_info")
              val infoDORQRErrorTest = (extraErrorMsg: String) => (infoDORGQRResult cne 0)
                .orEmpty(Code._fatalWithID[Unit](const(s"LAPACK error DORGQR. $extraErrorMsg Error code = ").concat(infoDORGQRResult.toS), errorID))

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
      case x: NDArrayMap => emitDeforestedNDArrayI(x)
      case x: NDArrayMap2 => emitDeforestedNDArrayI(x)
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

        val pt = PCanonicalTuple(false, sig.map(_.pResultType): _*)

        val addr = cb.newLocal("resultop_tuple_addr", pt.allocate(region))
        cb += pt.stagedInitialize(addr, setMissing = false)

        (0 until aggs.length).foreach { j =>
          val idx = start + j
          val rvAgg = agg.Extract.getAgg(sig(j))
          val fieldAddr = cb.newLocal(s"resultop_field_addr_$j", pt.fieldOffset(addr, j))
          rvAgg.storeResult(cb, sc.states(idx), pt.types(j), fieldAddr, region,
            (cb: EmitCodeBuilder) => cb += pt.setFieldMissing(addr, j))
        }

        presentPC(pt.loadCheapSCode(cb, addr))
      case x@ApplySeeded(fn, args, seed, rt) =>
        val codeArgs = args.map(a => EmitCode.fromI(cb.emb)(emitInNewBuilder(_, a)))
        val impl = x.implementation
        val unified = impl.unify(Array.empty[Type], args.map(_.typ), rt)
        assert(unified)
        impl.applySeededI(seed, cb, region, impl.computeReturnEmitType(x.typ, codeArgs.map(_.emitType)).st, codeArgs: _*)

      case AggStateValue(i, _) =>
        val AggContainer(_, sc, _) = container.get
        presentPC(sc.states(i).serializeToRegion(cb, PCanonicalBinary(), region))

      case ToArray(a) =>
        EmitStream.produce(this, a, cb, region, env, container)
          .map(cb) { case stream: SStreamCode => StreamUtils.toArray(cb, stream.producer, region) }

      case x@StreamFold(a, zero, accumName, valueName, body) =>
        EmitStream.produce(this, a, cb, region, env, container)
          .flatMap(cb) { case (stream: SStreamCode) =>
            val producer = stream.producer

            val stateEmitType = VirtualTypeWithReq(zero.typ, ctx.req.lookupState(x).head.asInstanceOf[TypeWithRequiredness]).canonicalEmitType

            val xAcc = mb.newEmitField(accumName, stateEmitType)
            val xElt = mb.newEmitField(valueName, producer.element.emitType)

            var tmpRegion: Settable[Region] = null

            if (producer.requiresMemoryManagementPerElement) {
              cb.assign(producer.elementRegion, Region.stagedCreate(Region.REGULAR, region.getPool()))

              tmpRegion = mb.genFieldThisRef[Region]("streamfold_tmpregion")
              cb.assign(tmpRegion, Region.stagedCreate(Region.REGULAR, region.getPool()))

              cb.assign(xAcc, emitI(zero, tmpRegion)
                .map(cb)(pc => pc.castTo(cb, tmpRegion, stateEmitType.st)))
            } else {
              cb.assign(producer.elementRegion, region)
              cb.assign(xAcc, emitI(zero, producer.elementRegion)
                .map(cb)(pc => pc.castTo(cb, producer.elementRegion, stateEmitType.st)))
            }

            producer.unmanagedConsume(cb) { cb =>
              cb.assign(xElt, producer.element)

              if (producer.requiresMemoryManagementPerElement) {
                cb.assign(xAcc, emitI(body, producer.elementRegion, env.bind(accumName -> xAcc, valueName -> xElt))
                  .map(cb)(pc => pc.castTo(cb, tmpRegion, stateEmitType.st, deepCopy = true)))
                cb += producer.elementRegion.clearRegion()
                val swapRegion = cb.newLocal[Region]("streamfold_swap_region", producer.elementRegion)
                cb.assign(producer.elementRegion, tmpRegion.load())
                cb.assign(tmpRegion, swapRegion.load())
              } else {
                cb.assign(xAcc, emitI(body, producer.elementRegion, env.bind(accumName -> xAcc, valueName -> xElt))
                  .map(cb)(pc => pc.castTo(cb, producer.elementRegion, stateEmitType.st, deepCopy = false)))
              }
            }

            if (producer.requiresMemoryManagementPerElement) {
              cb.assign(xAcc, xAcc.toI(cb).map(cb)(pc => pc.castTo(cb, region, pc.st, deepCopy = true)))
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

            val accTypes = ctx.req.lookupState(x).zip(acc.map(_._2.typ))
              .map { case (btwr, t) => VirtualTypeWithReq(t, btwr.asInstanceOf[TypeWithRequiredness])
                .canonicalEmitType
              }

            val xElt = mb.newEmitField(valueName, producer.element.emitType)
            val names = acc.map(_._1)
            val accVars = (names, accTypes).zipped.map(mb.newEmitField)

            val resEnv = env.bind(names.zip(accVars): _*)
            val seqEnv = resEnv.bind(valueName, xElt)

            if (producer.requiresMemoryManagementPerElement) {
              cb.assign(producer.elementRegion, Region.stagedCreate(Region.REGULAR, region.getPool()))

              tmpRegion = mb.genFieldThisRef[Region]("streamfold_tmpregion")
              cb.assign(tmpRegion, Region.stagedCreate(Region.REGULAR, region.getPool()))

              (accVars, acc).zipped.foreach { case (xAcc, (_, x)) =>
                cb.assign(xAcc, emitI(x, tmpRegion).map(cb)(_.castTo(cb, tmpRegion, xAcc.st)))
              }
            } else {
              cb.assign(producer.elementRegion, region)
              (accVars, acc).zipped.foreach { case (xAcc, (_, x)) =>
                cb.assign(xAcc, emitI(x, region).map(cb)(_.castTo(cb, region, xAcc.st)))
              }
            }

            producer.unmanagedConsume(cb) { cb =>
              cb.assign(xElt, producer.element)
              if (producer.requiresMemoryManagementPerElement) {
                (accVars, seq).zipped.foreach { (accVar, ir) =>
                  cb.assign(accVar,
                    emitI(ir, producer.elementRegion, env = seqEnv)
                      .map(cb)(pc => pc.castTo(cb, tmpRegion, accVar.st, deepCopy = true)))
                }
                cb += producer.elementRegion.clearRegion()
                val swapRegion = cb.newLocal[Region]("streamfold2_swap_region", producer.elementRegion)
                cb.assign(producer.elementRegion, tmpRegion.load())
                cb.assign(tmpRegion, swapRegion.load())
              } else {
                (accVars, seq).zipped.foreach { (accVar, ir) =>
                  cb.assign(accVar,
                    emitI(ir, producer.elementRegion, env = seqEnv)
                      .map(cb)(pc => pc.castTo(cb, producer.elementRegion, accVar.st, deepCopy = false)))
                }
              }
            }

            if (producer.requiresMemoryManagementPerElement) {
              accVars.foreach { xAcc =>
                cb.assign(xAcc, xAcc.toI(cb).map(cb)(pc => pc.castTo(cb, region, pc.st, deepCopy = true)))
              }
              cb += producer.elementRegion.invalidate()
              cb += tmpRegion.invalidate()
            }
            emitI(res, env = resEnv)
          }
      case t@Trap(child) =>
        val (ev, mb) = emitSplitMethod("trap", cb, child, region, env, container, loopEnv)
        val maybeException = cb.newLocal[(String, java.lang.Integer)]("trap_msg", cb.emb.ecb.runMethodWithHailExceptionHandler(mb.mb.methodName))
        val sst = SStringPointer(PCanonicalString(false))

        val tt = t.typ.asInstanceOf[TTuple]
        val errTupleType = tt.types(0).asInstanceOf[TTuple]
        val errTuple = SStackStruct(errTupleType, FastIndexedSeq(EmitType(sst, true), EmitType(SInt32, true)))
        val tv = cb.emb.newEmitField("trap_errTuple", EmitType(errTuple, false))

        val maybeMissingEV = cb.emb.newEmitField("trap_value", ev.emitType.copy(required = false))
        cb.ifx(maybeException.isNull, {
          cb.assign(tv, EmitCode.missing(cb.emb, errTuple))
          cb.assign(maybeMissingEV, ev)
        }, {
          val str = EmitCode.fromI(mb)(cb => IEmitCode.present(cb, sst.constructFromString(cb, region, maybeException.invoke[String]("_1"))))
          val errorId = EmitCode.fromI(mb)(cb =>
            IEmitCode.present(cb, primitive(maybeException.invoke[java.lang.Integer]("_2").invoke[Int]("intValue"))))
          cb.assign(tv, IEmitCode.present(cb, SStackStruct.constructFromArgs(cb, region, errTupleType, str, errorId)))
          cb.assign(maybeMissingEV, EmitCode.missing(cb.emb, ev.st))
        })
        IEmitCode.present(cb, {
          SStackStruct.constructFromArgs(cb, region, t.typ.asInstanceOf[TBaseStruct], tv, maybeMissingEV)
        })

      case Die(m, typ, errorId) =>
        val cm = emitI(m)
        val msg = cb.newLocal[String]("die_msg")
        cm.consume(cb,
          cb.assign(msg, "<exception message missing>"),
          { sc => cb.assign(msg, sc.asString.loadString()) })
        cb._throw[HailException](Code.newInstance[HailException, String, Int](msg, errorId))

        IEmitCode.present(cb, SUnreachable.fromVirtualType(typ).defaultValue)

      case CastToArray(a) =>
        emitI(a).map(cb) { ind => ind.asIndexable.castToArray(cb) }

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

        val shuffleST = SCanonicalShufflePointer(PCanonicalShuffle(shuffleType, false))
        val settable = mb.newPField(shuffleST).asInstanceOf[SCanonicalShufflePointerSettable]
        val shuffle = CompileTimeShuffleClient.create(cb, settable)

        shuffle.start(cb, region)

        val shuffleEnv = env.bind(name -> EmitSettable.present(settable))

        val successfulShuffleIds: SValue = emitI(writerIR, env = shuffleEnv)
          .get(cb, "shuffle ID must be non-missing")
          // just store it so the writer gets run
          .memoize(cb, "shuffleSuccessfulShuffleIds")

        val shuffleReaders =
          emitI(readersIR, env = shuffleEnv).memoize(cb, "shuffleReaders")

        shuffle.stop(cb)
        shuffle.close(cb)

        shuffleReaders.toI(cb)
      case ShuffleWrite(idIR, rowsIR) =>
        val rows = emitStream(rowsIR, cb, region)
          .get(cb, "rows stream was missing in shuffle write")
          .asStream

        val uuid = emitI(idIR)
          .get(cb, "shuffle ID must be non-missing")
          .asInstanceOf[SCanonicalShufflePointerCode]
          .memoize(cb, "shuffleClientUUID")

        val shuffle = CompileTimeShuffleClient.create(cb, uuid)
        shuffle.startPut(cb)

        val producer = rows.producer
        producer.memoryManagedConsume(region, cb) { cb =>
          producer.element.toI(cb).consume(cb,
            cb._fatal(s"empty row in shuffle put"),
            { sc => shuffle.putValue(cb, sc) })
        }

        shuffle.finishPut(cb)
        shuffle.close(cb)

        val resPType = PCanonicalBinary()
        // FIXME: server needs to send uuid for the successful partition
        val boff = cb.memoize(resPType.loadCheapSCode(cb, resPType.allocate(region, 0)), "shuffleWriteBOff")
        val baddr = SingleCodeSCode.fromSCode(cb, boff, region)
        cb += resPType.storeLength(baddr.code.asInstanceOf[Code[Long]], 0)
        presentPC(boff)

      case x@ReadValue(path, spec, requestedType) =>
        emitI(path).map(cb) { pv =>
          val ib = cb.newLocal[InputBuffer]("read_ib")
          cb.assign(ib, spec.buildCodeInputBuffer(mb.open(pv.asString.loadString(), checkCodec = true)))
          spec.encodedType.buildDecoder(requestedType, mb.ecb)(cb, region, ib)
        }

      case WriteValue(value, path, spec) =>
        emitI(path).flatMap(cb) { case p: SStringCode =>
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

        val accTypes = ctx.req.lookupState(x).zip(args.map(_._2.typ))
          .map { case (btwr, t) => VirtualTypeWithReq(t, btwr.asInstanceOf[TypeWithRequiredness])
            .canonicalEmitType
          }

        val inits = args.zip(accTypes)

        val stagedPool = cb.newLocal[RegionPool]("tail_loop_pool_ref")
        cb.assign(stagedPool, region.getPool())

        val resultEmitType = ctx.req.lookup(body).asInstanceOf[TypeWithRequiredness].canonicalEmitType(body.typ)
        val loopRef = LoopRef(cb, loopStartLabel, inits.map { case ((name, _), pt) => (name, pt) }, stagedPool, resultEmitType)

        val argEnv = env
          .bind((args.map(_._1), loopRef.loopArgs).zipped.toArray: _*)

        val newLoopEnv = loopEnv.getOrElse(Env.empty)

        // Emit into LoopRef's current region. (region 1)
        loopRef.loopArgs.zip(inits).foreach { case (settable, ((_, x), et)) =>
          settable.store(cb, emitI(x, loopRef.r1).map(cb)(_.castTo(cb, loopRef.r1, et.st)))
        }

        cb.define(loopStartLabel)

        val result = emitI(body, region=loopRef.r1, env = argEnv, loopEnv = Some(newLoopEnv.bind(name, loopRef))).map(cb) { pc =>
          val answerInRightRegion = pc.copyToRegion(cb, region, pc.st)
          cb.append(loopRef.r1.clearRegion())
          cb.append(loopRef.r2.clearRegion())
          answerInRightRegion
        }
        assert(result.emitType == resultEmitType, s"loop type mismatch: emitted=${ result.emitType }, expected=${ resultEmitType }")
        result

      case Recur(name, args, _) =>
        val loopRef = loopEnv.get.lookup(name)

        // Need to emit into region 1, copy to region 2, then clear region 1, then swap them.
        (loopRef.tmpLoopArgs, loopRef.loopTypes, args).zipped.map { case (tmpLoopArg, et, arg) =>
          tmpLoopArg.store(cb, emitI(arg, loopEnv = None, region = loopRef.r1).map(cb)(_.copyToRegion(cb, loopRef.r2, et.st)))
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

        val rt = loopRef.resultType
        IEmitCode(CodeLabel(), CodeLabel(), rt.st.defaultValue, rt.required)

      case x@CollectDistributedArray(contexts, globals, cname, gname, body, tsd) =>
        val parentCB = mb.ecb
        emitStream(contexts, cb, region).map(cb) { case ctxStream: SStreamCode =>

          def wrapInTuple(cb: EmitCodeBuilder, region: Value[Region], et: EmitCode): SBaseStructPointerCode = {
            PCanonicalTuple(true, et.emitType.canonicalPType).constructFromFields(cb, region, FastIndexedSeq(et), deepCopy = false)
          }

          val bufferSpec: BufferSpec = BufferSpec.defaultUncompressed

          val emitGlobals = EmitCode.fromI(mb)(cb => emitInNewBuilder(cb, globals))

          val ctxType = ctxStream.st.elementEmitType
          val contextPTuple: PTuple = PCanonicalTuple(required = true, ctxType.canonicalPType)
          val globalPTuple: PTuple = PCanonicalTuple(required = true, emitGlobals.emitType.canonicalPType)
          val contextSpec: TypedCodecSpec = TypedCodecSpec(contextPTuple, bufferSpec)
          val globalSpec: TypedCodecSpec = TypedCodecSpec(globalPTuple, bufferSpec)

          // emit body in new FB
          val bodyFB = EmitFunctionBuilder[Region, Array[Byte], Array[Byte], Array[Byte]](ctx.executeContext, "collect_distributed_array")

          var bodySpec: TypedCodecSpec = null
          bodyFB.emitWithBuilder { cb =>
            val region = bodyFB.getCodeParam[Region](1)
            val ctxIB = cb.newLocal[InputBuffer]("cda_ctx_ib", contextSpec.buildCodeInputBuffer(
              Code.newInstance[ByteArrayInputStream, Array[Byte]](bodyFB.getCodeParam[Array[Byte]](2))))
            val gIB = cb.newLocal[InputBuffer]("cda_g_ib", globalSpec.buildCodeInputBuffer(
              Code.newInstance[ByteArrayInputStream, Array[Byte]](bodyFB.getCodeParam[Array[Byte]](3))))

            val decodedContext = contextSpec.encodedType.buildDecoder(contextSpec.encodedVirtualType, bodyFB.ecb)
              .apply(cb, region, ctxIB)
              .asBaseStruct
              .memoize(cb, "decoded_context_tuple")
              .loadField(cb, 0)
              .memoizeField(cb, "decoded_context")

            val decodedGlobal = globalSpec.encodedType.buildDecoder(globalSpec.encodedVirtualType, bodyFB.ecb)
              .apply(cb, region, gIB)
              .asBaseStruct
              .memoize(cb, "decoded_global_tuple")
              .loadField(cb, 0)
              .memoizeField(cb, "decoded_global")

            val env = EmitEnv(Env[EmitValue](
              (cname, decodedContext),
              (gname, decodedGlobal)), FastIndexedSeq())

            val bodyResult = wrapInTuple(cb,
              region,
              EmitCode.fromI(cb.emb)(cb => new Emit(ctx, bodyFB.ecb).emitI(body, cb, env, None)))

            bodySpec = TypedCodecSpec(bodyResult.st.canonicalPType().setRequired(true), bufferSpec)

            val bOS = cb.newLocal[ByteArrayOutputStream]("cda_baos", Code.newInstance[ByteArrayOutputStream]())
            val bOB = cb.newLocal[OutputBuffer]("cda_ob", bodySpec.buildCodeOutputBuffer(bOS))
            bodySpec.encodedType.buildEncoder(bodyResult.st, cb.emb.ecb)
              .apply(cb, bodyResult, bOB)
            cb += bOB.invoke[Unit]("flush")
            cb += bOB.invoke[Unit]("close")
            bOS.invoke[Array[Byte]]("toByteArray")
          }

          val fID = genUID()
          parentCB.addModule(fID, bodyFB.resultWithIndex())
          val functionID = fID

          val spark = parentCB.backend()

          val baos = mb.genFieldThisRef[ByteArrayOutputStream]()
          val buf = mb.genFieldThisRef[OutputBuffer]()
          val ctxab = mb.genFieldThisRef[ByteArrayArrayBuilder]()
          val encRes = mb.genFieldThisRef[Array[Array[Byte]]]()


          def addContexts(cb: EmitCodeBuilder, ctxStream: StreamProducer): Unit = {
            ctxStream.memoryManagedConsume(region, cb, setup = { cb =>
              cb += ctxab.invoke[Int, Unit]("ensureCapacity", ctxStream.length.map(_.apply(cb)).getOrElse(16))
            }) { cb =>
              cb += baos.invoke[Unit]("reset")
              val ctxTuple = wrapInTuple(cb, region, ctxStream.element)
                .memoize(cb, "cda_add_contexts_addr")
              contextSpec.encodedType.buildEncoder(ctxTuple.st, parentCB)
                .apply(cb, ctxTuple, buf)
              cb += buf.invoke[Unit]("flush")
              cb += ctxab.invoke[Array[Byte], Unit]("add", baos.invoke[Array[Byte]]("toByteArray"))
            }
          }

          def addGlobals(cb: EmitCodeBuilder): Unit = {
            val wrapped = wrapInTuple(cb, region, emitGlobals)
            globalSpec.encodedType.buildEncoder(wrapped.st, parentCB)
              .apply(cb, wrapped, buf)
            cb += buf.invoke[Unit]("flush")
          }

          def decodeResult(cb: EmitCodeBuilder): SCode = {
            val len = mb.newLocal[Int]("cda_result_length")
            val ib = mb.newLocal[InputBuffer]("decode_ib")

            cb.assign(len, encRes.length())
            val pt = PCanonicalArray(bodySpec.encodedType.decodedSType(bodySpec.encodedVirtualType).asInstanceOf[SBaseStruct].fieldEmitTypes(0).canonicalPType)
            pt.asInstanceOf[PCanonicalArray].constructFromElements(cb, region, len, deepCopy = false) { (cb, i) =>
              cb.assign(ib, bodySpec.buildCodeInputBuffer(Code.newInstance[ByteArrayInputStream, Array[Byte]](encRes(i))))
              val eltTupled = bodySpec.encodedType.buildDecoder(bodySpec.encodedVirtualType, parentCB)
                .apply(cb, region, ib)
                .asBaseStruct
                .memoize(cb, "cda_eltTupled")
              eltTupled.loadField(cb, 0)
            }
          }

          cb.assign(baos, Code.newInstance[ByteArrayOutputStream]())
          cb.assign(buf, contextSpec.buildCodeOutputBuffer(baos)) // TODO: take a closer look at whether we need two codec buffers?
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

    ctx.req.lookupOpt(ir) match {
      case Some(r) =>
        if (result.required != r.required) {
          throw new RuntimeException(s"requiredness mismatch: EC=${ result.required } / Analysis=${ r.required }\n${ result.st }\n${ Pretty(ir) }")
        }

      case _ =>
      // we dynamically generate some IRs in emission. Ignore these...
    }

    if (result.st.virtualType != ir.typ)
      throw new RuntimeException(s"type mismatch:\n  EC=${ result.st.virtualType }\n  IR=${ ir.typ }\n  node: ${ Pretty(ir).take(50) }")

    result
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
  private[ir] def emit(ir: IR, mb: EmitMethodBuilder[C], env: EmitEnv, container: Option[AggContainer]): EmitCode = {
    val region = mb.getCodeParam[Region](1)
    emit(ir, mb, region, env, container, None)
  }

  private[ir] def emitWithRegion(ir: IR, mb: EmitMethodBuilder[C], region: Value[Region], env: EmitEnv, container: Option[AggContainer]): EmitCode =
    emit(ir, mb, region, env, container, None)

  private def emit(
    ir: IR,
    mb: EmitMethodBuilder[C],
    region: Value[Region],
    env: EmitEnv,
    container: Option[AggContainer],
    loopEnv: Option[Env[LoopRef]],
    fallingBackFromEmitI: Boolean = false
  ): EmitCode = {

    if (ctx.methodSplits.contains(ir) && !ctx.tryingToSplit.contains(ir)) {
      return EmitCode.fromI(mb)(cb => emitInSeparateMethod(s"split_${ir.getClass.getSimpleName}", cb, ir, region, env, container, loopEnv))
    }


    def emit(ir: IR, region: Value[Region] = region, env: EmitEnv = env, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): EmitCode =
      this.emit(ir, mb, region, env, container, loopEnv)

    def emitI(ir: IR, cb: EmitCodeBuilder, env: EmitEnv = env, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): IEmitCode =
      this.emitI(ir, cb, region, env, container, loopEnv)

    def emitVoid(ir: IR, env: EmitEnv = env, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): Code[Unit] = {
      EmitCodeBuilder.scopedVoid(mb) { cb =>
        this.emitVoid(cb, ir, region, env, container, loopEnv)
      }
    }

    def emitStream(ir: IR, outerRegion: Value[Region]): EmitCode =
      EmitCode.fromI(mb)(cb => EmitStream.produce(this, ir, cb, outerRegion, env, container))

    // ideally, emit would not be called with void values, but initOp args can be void
    // working towards removing this
    if (ir.typ == TVoid)
      return EmitCode.fromI(mb) { cb =>
        this.emitVoid(cb, ir, region, env, container, loopEnv)
        IEmitCode.present(cb, SCode._empty)
      }

    val result: EmitCode = (ir: @unchecked) match {

      case Let(name, value, body) =>
        EmitCode.fromI(mb) { cb =>
          val xVal = if (value.typ.isInstanceOf[TStream]) emitStream(value, region) else emit(value)
          cb.withScopedMaybeStreamValue(xVal, s"let_$name") { ev =>
            emitI(body, cb, env = env.bind(name, ev))
          }
        }

      case Ref(name, t) =>
        val ev = env.bindings.lookup(name)
        if (ev.st.virtualType != t)
          throw new RuntimeException(s"emit value type did not match specified type:\n name: $name\n  ev: ${ ev.st.virtualType }\n  ir: ${ ir.typ }")
        ev.load

      case In(i, expectedPType) =>
        // this, Code[Region], ...
        val ev = env.inputValues(i).apply(region)
        ev

      case ir@Apply(fn, typeArgs, args, rt, errorID) =>
        val impl = ir.implementation
        val unified = impl.unify(typeArgs, args.map(_.typ), rt)
        assert(unified)

        val emitArgs = args.map(a => EmitCode.fromI(mb)(emitI(a, _))).toFastIndexedSeq

        val argSTypes = emitArgs.map(_.st)
        val retType = impl.computeStrictReturnEmitType(ir.typ, argSTypes)
        val k = (fn, typeArgs, argSTypes, retType)
        val meth =
          methods.get(k) match {
            case Some(funcMB) =>
              funcMB
            case None =>
              val funcMB = impl.getAsMethod(mb.ecb, retType, typeArgs, argSTypes: _*)
              methods.update(k, funcMB)
              funcMB
          }
        EmitCode.fromI(mb) { cb =>
          val emitArgs = args.map(a => EmitCode.fromI(cb.emb)(emitI(a, _))).toFastIndexedSeq
          IEmitCode.multiMapEmitCodes(cb, emitArgs) { codeArgs =>
            cb.invokeSCode(meth, FastIndexedSeq[Param](CodeParam(region), CodeParam(errorID)) ++ codeArgs.map(pc => pc: Param): _*)
          }
        }
      case x@ApplySpecial(_, typeArgs, args, rt, errorID) =>
        val codeArgs = args.map(a => emit(a))
        val impl = x.implementation
        val unified = impl.unify(typeArgs, args.map(_.typ), rt)
        assert(unified)
        val retType = impl.computeReturnEmitType(x.typ, codeArgs.map(_.emitType))
        impl.apply(EmitRegion(mb, region), retType.st, typeArgs, errorID, codeArgs: _*)

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

    ctx.req.lookupOpt(ir) match {
      case Some(r) =>
        if (result.required != r.required) {
          throw new RuntimeException(s"requiredness mismatch: EC=${ result.required } / Analysis=${ r.required }\n${ result.emitType }\n${ Pretty(ir) }")
        }

      case _ =>
      // we dynamically generate some IRs in emission. Ignore these...
    }

    if (result.st.virtualType != ir.typ)
      throw new RuntimeException(s"type mismatch: EC=${ result.st.virtualType } / IR=${ ir.typ }\n${ ir.getClass.getSimpleName }")
    result
  }

  private def makeDependentSortingFunction(
    cb: EmitCodeBuilder,
    elemSCT: SingleCodeType, ir: IR, env: EmitEnv, emitter: Emit[_], leftRightComparatorNames: Array[String]): (EmitCodeBuilder, Value[Region], Code[_], Code[_]) => Code[Boolean] = {
    val fb = cb.emb.ecb

    var newEnv = env
    val sort = fb.genEmitMethod("dependent_sorting_func",
      FastIndexedSeq(typeInfo[Region], CodeParamType(elemSCT.ti), CodeParamType(elemSCT.ti)),
      BooleanInfo)

    sort.emitWithBuilder[Boolean] { cb =>
      val region = sort.getCodeParam[Region](1)
      val leftEC = cb.memoize(EmitCode.present(sort, elemSCT.loadToSCode(cb, region, sort.getCodeParam(2)(elemSCT.ti))), "sort_leftEC")
      val rightEC = cb.memoize(EmitCode.present(sort, elemSCT.loadToSCode(cb, region, sort.getCodeParam(3)(elemSCT.ti))), "sort_rightEC")

      if (leftRightComparatorNames.nonEmpty) {
        assert(leftRightComparatorNames.length == 2)
        newEnv = newEnv.bind(
            (leftRightComparatorNames(0), leftEC),
            (leftRightComparatorNames(1), rightEC))
      }

      val iec = emitter.emitI(ir, cb, newEnv, None)
      iec.get(cb, "Result of sorting function cannot be missing").asBoolean.boolCode(cb)
    }
    (cb: EmitCodeBuilder, region: Value[Region], l: Code[_], r: Code[_]) => cb.invokeCode[Boolean](sort, region, l, r)
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
    }
    }
  }

  def unifyShapes2(cb: EmitCodeBuilder, leftShape: IndexedSeq[Value[Long]], rightShape: IndexedSeq[Value[Long]], errorID: Int): IndexedSeq[Value[Long]] = {
    val shape = leftShape.zip(rightShape).zipWithIndex.map { case ((left, right), i) =>
      val notSameAndNotBroadcastable = !((left ceq right) || (left ceq 1L) || (right ceq 1L))
      cb.newField[Long](
        s"unify_shapes2_shape$i",
        notSameAndNotBroadcastable.mux(
          Code._fatalWithID[Long](rightShape.foldLeft[Code[String]](
            leftShape.foldLeft[Code[String]](
              const("Incompatible NDArray shapes: [ ")
            )((accum, v) => accum.concat(v.toS).concat(" "))
              .concat("] vs [ ")
          )((accum, v) => accum.concat(v.toS).concat(" "))
            .concat("]"), errorID),
          (left > right).mux(left, right)))
    }

    shape
  }

  def matmulShape(cb: EmitCodeBuilder, leftShape: IndexedSeq[Value[Long]], rightShape: IndexedSeq[Value[Long]], errorID: Int): IndexedSeq[Value[Long]] = {
    val mb = cb.emb

    assert(leftShape.nonEmpty)
    assert(rightShape.nonEmpty)

    var lK: Value[Long] = null
    var rK: Value[Long] = null
    var shape: IndexedSeq[Value[Long]] = null

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
        val unifiedShape = unifyShapes2(cb,
          leftShape.slice(0, leftShape.length - 2),
          rightShape.slice(0, rightShape.length - 2), errorID)
        shape = unifiedShape :+ leftShape(leftShape.length - 2) :+ rightShape.last
      }
    }

    val leftShapeString = const("(").concat(leftShape.map(_.toS).reduce((a, b) => a.concat(", ").concat(b))).concat(")")
    val rightShapeString = const("(").concat(rightShape.map(_.toS).reduce((a, b) => a.concat(", ").concat(b))).concat(")")


    cb.ifx(lK.cne(rK), {
      cb._fatalWithError(errorID,"Matrix dimensions incompatible: ", leftShapeString,
        " can't be multiplied by matrix with dimensions ", rightShapeString)
    })

    shape
  }
}

abstract class NDArrayEmitter(val outputShape: IndexedSeq[Value[Long]], val elementType: SType) {
  val nDims = outputShape.length

  def outputElement(cb: EmitCodeBuilder, idxVars: IndexedSeq[Value[Long]]): SCode

  def emit(cb: EmitCodeBuilder, targetType: PCanonicalNDArray, region: Value[Region]): SCode = {
    val shapeArray = outputShape

    val idx = cb.newLocal[Int]("ndarrayemitter_emitloops_idx", 0)

    val (firstElementAddress, finish) = targetType.constructDataFunction(
      outputShape,
      targetType.makeColumnMajorStrides(shapeArray, region, cb),
      cb,
      region)

    SNDArray.forEachIndexColMajor(cb, shapeArray, "ndarrayemitter_emitloops") { case (cb, idxVars) =>
      val element = IEmitCode.present(cb, outputElement(cb, idxVars)).consume(cb, {
        cb._fatal("NDArray elements cannot be missing")
      }, { elementPc =>
        targetType.elementType.storeAtAddress(cb, firstElementAddress + (idx.toL * targetType.elementType.byteSize), region, elementPc, true)
      })
      cb.assign(idx, idx + 1)
    }

    finish(cb)
  }
}

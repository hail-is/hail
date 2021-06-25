package is.hail.expr.ir

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir.agg.AggStateSig
import is.hail.expr.ir.lowering.LoweringPipeline
import is.hail.expr.ir.streams.{EmitStream, StreamArgType}
import is.hail.io.fs.FS
import is.hail.rvd.RVDContext
import is.hail.types.physical.stypes.{PTypeReferenceSingleCodeType, SingleCodeType, StreamSingleCodeType}
import is.hail.types.physical.stypes.interfaces.SStream
import is.hail.types.physical.{PStream, PStruct, PType}
import is.hail.types.virtual.Type
import is.hail.utils._

import java.io.PrintWriter

case class CodeCacheKey(aggSigs: IndexedSeq[AggStateSig], args: Seq[(String, EmitParamType)], body: IR)

case class CodeCacheValue(typ: Option[SingleCodeType], f: (FS, Int, Region) => Any)

object Compile {
  private[this] val codeCache: Cache[CodeCacheKey, CodeCacheValue] = new Cache(50)

  def apply[F: TypeInfo](
    ctx: ExecuteContext,
    params: IndexedSeq[(String, EmitParamType)],
    expectedCodeParamTypes: IndexedSeq[TypeInfo[_]], expectedCodeReturnType: TypeInfo[_],
    body: IR,
    optimize: Boolean = true,
    print: Option[PrintWriter] = None
  ): (Option[SingleCodeType], (FS, Int, Region) => F) = {

    val normalizeNames = new NormalizeNames(_.toString)
    val normalizedBody = normalizeNames(body,
      Env(params.map { case (n, _) => n -> n }: _*))
    val k = CodeCacheKey(FastIndexedSeq[AggStateSig](), params.map { case (n, pt) => (n, pt) }, normalizedBody)
    codeCache.get(k) match {
      case Some(v) =>
        return (v.typ, v.f.asInstanceOf[(FS, Int, Region) => F])
      case None =>
    }

    var ir = body
    ir = Subst(ir, BindingEnv(params
      .zipWithIndex
      .foldLeft(Env.empty[IR]) { case (e, ((n, t), i)) => e.bind(n, In(i, t)) }))
    ir = LoweringPipeline.compileLowerer(optimize).apply(ctx, ir).asInstanceOf[IR].noSharing

    TypeCheck(ir, BindingEnv.empty)

    val returnParam = CodeParamType(SingleCodeType.typeInfoFromType(ir.typ))

    val fb = EmitFunctionBuilder[F](ctx, "Compiled",
      CodeParamType(typeInfo[Region]) +: params.map { case (_, pt) =>
        pt
      }, returnParam, Some("Emit.scala"))

    /*
    {
      def visit(x: IR): Unit = {
        println(f"${ System.identityHashCode(x) }%08x    ${ x.getClass.getSimpleName } ${ x.pType }")
        Children(x).foreach {
          case c: IR => visit(c)
        }
      }

      visit(ir)
    }
     */

    assert(fb.mb.parameterTypeInfo == expectedCodeParamTypes, s"expected $expectedCodeParamTypes, got ${ fb.mb.parameterTypeInfo }")
    assert(fb.mb.returnTypeInfo == expectedCodeReturnType, s"expected $expectedCodeReturnType, got ${ fb.mb.returnTypeInfo }")

    val emitContext = EmitContext.analyze(ctx, ir)
    val rt = Emit(emitContext, ir, fb, expectedCodeReturnType, params.length)

    val f = fb.resultWithIndex(print)
    codeCache += k -> CodeCacheValue(rt, f)

    (rt, f)
  }
}

object CompileWithAggregators {
  private[this] val codeCache: Cache[CodeCacheKey, CodeCacheValue] = new Cache(50)

  def apply[F: TypeInfo](
    ctx: ExecuteContext,
    aggSigs: Array[AggStateSig],
    params: IndexedSeq[(String, EmitParamType)],
    expectedCodeParamTypes: IndexedSeq[TypeInfo[_]], expectedCodeReturnType: TypeInfo[_],
    body: IR,
    optimize: Boolean = true
  ): (Option[SingleCodeType], (FS, Int, Region) => (F with FunctionWithAggRegion)) = {
    val normalizeNames = new NormalizeNames(_.toString)
    val normalizedBody = normalizeNames(body,
      Env(params.map { case (n, _) => n -> n }: _*))
    val k = CodeCacheKey(aggSigs, params.map { case (n, pt) => (n, pt) }, normalizedBody)
    codeCache.get(k) match {
      case Some(v) =>
        return (v.typ, v.f.asInstanceOf[(FS, Int, Region) => (F with FunctionWithAggRegion)])
      case None =>
    }

    var ir = body
    ir = Subst(ir, BindingEnv(params
      .zipWithIndex
      .foldLeft(Env.empty[IR]) { case (e, ((n, t), i)) => e.bind(n, In(i, t)) }))
    ir = LoweringPipeline.compileLowerer(optimize).apply(ctx, ir).asInstanceOf[IR].noSharing

    TypeCheck(ir, BindingEnv(Env.fromSeq[Type](params.map { case (name, t) => name -> t.virtualType })))

    val fb = EmitFunctionBuilder[F](ctx, "CompiledWithAggs",
      CodeParamType(typeInfo[Region]) +: params.map { case (_, pt) => pt },
      SingleCodeType.typeInfoFromType(ir.typ), Some("Emit.scala"))

    /*
    {
      def visit(x: IR): Unit = {
        println(f"${ System.identityHashCode(x) }%08x    ${ x.getClass.getSimpleName } ${ x.pType }")
        Children(x).foreach {
          case c: IR => visit(c)
        }
      }

      visit(ir)
    }
     */

    val emitContext = EmitContext.analyze(ctx, ir)
    val rt = Emit(emitContext, ir, fb, expectedCodeReturnType, params.length, Some(aggSigs))

    val f = fb.resultWithIndex()
    codeCache += k -> CodeCacheValue(rt, f)
    (rt, f.asInstanceOf[(FS, Int, Region) => (F with FunctionWithAggRegion)])
  }
}

object CompileIterator {

  private trait StepFunctionBase {
    def loadAddress(): Long
  }

  private trait TableStageToRVDStepFunction extends StepFunctionBase {
    def apply(o: Object, a: Long, b: Long): Boolean

    def setRegions(outerRegion: Region, eltRegion: Region): Unit
  }

  private trait TMPStepFunction extends StepFunctionBase {
    def apply(o: Object, a: Long, b: StreamArgType): Boolean

    def setRegions(outerRegion: Region, eltRegion: Region): Unit
  }

  private abstract class LongIteratorWrapper extends Iterator[java.lang.Long] {
    def step(): Boolean

    protected val stepFunction: StepFunctionBase
    private var _stepped = false
    private var _hasNext = false

    def hasNext: Boolean = {
      if (!_stepped) {
        _hasNext = step()
        _stepped = true
      }
      _hasNext
    }

    def next(): java.lang.Long = {
      if (!hasNext)
        return Iterator.empty.next()
      _stepped = false
      stepFunction.loadAddress()
    }
  }

  private def compileStepper[F >: Null <: StepFunctionBase : TypeInfo](
    ctx: ExecuteContext,
    body: IR,
    argTypeInfo: Array[ParamType],
    printWriter: Option[PrintWriter]
  ): (PType, (FS, Int, Region) => F) = {

    val fb = EmitFunctionBuilder.apply[F](ctx, "stream", argTypeInfo.toFastIndexedSeq, CodeParamType(BooleanInfo))
    val outerRegionField = fb.genFieldThisRef[Region]("outerRegion")
    val eltRegionField = fb.genFieldThisRef[Region]("eltRegion")
    val setF = fb.newEmitMethod("setRegions", FastIndexedSeq(CodeParamType(typeInfo[Region]), CodeParamType(typeInfo[Region])), CodeParamType(typeInfo[Unit]))
    setF.emit(Code(outerRegionField := setF.getCodeParam[Region](1), eltRegionField := setF.getCodeParam[Region](2)))

    val stepF = fb.apply_method
    val stepFECB = stepF.ecb

    val outerRegion = outerRegionField

    val ir = LoweringPipeline.compileLowerer(true)(ctx, body).asInstanceOf[IR].noSharing
    TypeCheck(ir)

    var elementAddress: Settable[Long] = null
    var returnType: PType = null

    stepF.emitWithBuilder[Boolean] { cb =>
      val emitContext = EmitContext.analyze(ctx, ir)
      val emitter = new Emit(emitContext, stepFECB)

      val env = EmitEnv(Env.empty, argTypeInfo.indices.filter(i => argTypeInfo(i).isInstanceOf[EmitParamType]).map(i => stepF.storeEmitParam(i + 1, cb)))
      val optStream = EmitCode.fromI(stepF)(cb => EmitStream.produce(emitter, ir, cb, outerRegion, env, None))
      returnType = optStream.st.asInstanceOf[SStream].elementEmitType.canonicalPType.setRequired(true)
      val returnPType = optStream.st.asInstanceOf[SStream].elementType.canonicalPType()

      elementAddress = stepF.genFieldThisRef[Long]("elementAddr")

      val didSetup = stepF.genFieldThisRef[Boolean]("didSetup")
      stepF.cb.emitInit(didSetup := false)

      val eosField = stepF.genFieldThisRef[Boolean]("eos")

      val producer = optStream.pv.asStream.producer

      val ret = cb.newLocal[Boolean]("stepf_ret")
      val Lreturn = CodeLabel()

      cb.ifx(!didSetup, {
        optStream.toI(cb).get(cb) // handle missing, but bound stream producer above

        cb.assign(producer.elementRegion, eltRegionField)
        producer.initialize(cb)
        cb.assign(didSetup, true)
        cb.assign(eosField, false)
      })

      cb.ifx(eosField, {
        cb.assign(ret, false)
        cb.goto(Lreturn)
      })

      cb.goto(producer.LproduceElement)

      stepF.implementLabel(producer.LendOfStream) { cb =>
        producer.close(cb)
        cb.assign(eosField, true)
        cb.assign(ret, false)
        cb.goto(Lreturn)
      }

      stepF.implementLabel(producer.LproduceElementDone) { cb =>
        val pc = producer.element.toI(cb).get(cb)
        cb.assign(elementAddress, returnPType.store(cb, producer.elementRegion, pc, false))
        cb.assign(ret, true)
        cb.goto(Lreturn)
      }

      cb.define(Lreturn)
      ret
    }


    val getMB = fb.newEmitMethod("loadAddress", FastIndexedSeq(), LongInfo)
    getMB.emit(elementAddress.load())

    (returnType, fb.resultWithIndex(printWriter))
  }

  def forTableMapPartitions(
    ctx: ExecuteContext,
    typ0: PStruct, typ1: PStream,
    ir: IR
  ): (PType, (FS, Int, RVDContext, Long, streams.StreamArgType) => Iterator[java.lang.Long]) = {
    assert(typ0.required)
    assert(typ1.required)
    val (eltPType, makeStepper) = compileStepper[TMPStepFunction](
      ctx, ir,
      Array[ParamType](
        CodeParamType(typeInfo[Object]),
        SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(typ0)),
        SingleCodeEmitParamType(true, StreamSingleCodeType(true, typ1.elementType))),
      None)
    (eltPType, (fs, idx, consumerCtx, v0, part) => {
      val stepper = makeStepper(fs, idx, consumerCtx.partitionRegion)
      stepper.setRegions(consumerCtx.partitionRegion, consumerCtx.region)
      new LongIteratorWrapper {
        val stepFunction: TMPStepFunction = stepper

        def step(): Boolean = stepper.apply(null, v0, part)
      }
    })
  }

  def forTableStageToRVD(
    ctx: ExecuteContext,
    ctxType: PStruct, bcValsType: PType,
    ir: IR
  ): (PType, (FS, Int, RVDContext, Long, Long) => Iterator[java.lang.Long]) = {
    assert(ctxType.required)
    assert(bcValsType.required)
    val (eltPType, makeStepper) = compileStepper[TableStageToRVDStepFunction](
      ctx, ir,
      Array[ParamType](
        CodeParamType(typeInfo[Object]),
        SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(ctxType)),
        SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(bcValsType))),
      None)
    (eltPType, (fs, idx, consumerCtx, v0, v1) => {
      val stepper = makeStepper(fs, idx, consumerCtx.partitionRegion)
      stepper.setRegions(consumerCtx.partitionRegion, consumerCtx.region)
      new LongIteratorWrapper {
        val stepFunction: TableStageToRVDStepFunction = stepper

        def step(): Boolean = stepper.apply(null, v0, v1)
      }
    })
  }

}

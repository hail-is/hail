package is.hail.expr.ir

import java.io.PrintWriter
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir.Stream.Source
import is.hail.expr.ir.agg.AggStateSig
import is.hail.expr.ir.lowering.LoweringPipeline
import is.hail.rvd.RVDContext
import is.hail.types.physical.{PStream, PStruct, PType, PTypeReferenceSingleCodeType, SingleCodeType, StreamSingleCodeType}
import is.hail.types.virtual.Type
import is.hail.utils._

case class CodeCacheKey(aggSigs: IndexedSeq[AggStateSig], args: Seq[(String, EmitParamType)], body: IR)

case class CodeCacheValue(typ: Option[SingleCodeType], f: (Int, Region) => Any)

object Compile {
  private[this] val codeCache: Cache[CodeCacheKey, CodeCacheValue] = new Cache(50)

  def apply[F: TypeInfo](
    ctx: ExecuteContext,
    params: IndexedSeq[(String, EmitParamType)],
    expectedCodeParamTypes: IndexedSeq[TypeInfo[_]], expectedCodeReturnType: TypeInfo[_],
    body: IR,
    optimize: Boolean = true,
    print: Option[PrintWriter] = None
  ): (Option[SingleCodeType], (Int, Region) => F) = {

    val normalizeNames = new NormalizeNames(_.toString)
    val normalizedBody = normalizeNames(body,
      Env(params.map { case (n, _) => n -> n }: _*))
    val k = CodeCacheKey(FastIndexedSeq[AggStateSig](), params.map { case (n, pt) => (n, pt) }, normalizedBody)
    codeCache.get(k) match {
      case Some(v) =>
        return (v.typ, v.f.asInstanceOf[(Int, Region) => F])
      case None =>
    }

    var ir = body
    ir = Subst(ir, BindingEnv(params
      .zipWithIndex
      .foldLeft(Env.empty[IR]) { case (e, ((n, t), i)) => e.bind(n, In(i, t)) }))
    ir = LoweringPipeline.compileLowerer(optimize).apply(ctx, ir).asInstanceOf[IR].noSharing

    TypeCheck(ir, BindingEnv.empty)

    val usesAndDefs = ComputeUsesAndDefs(ir, errorIfFreeVariables = false)
    val requiredness = Requiredness.apply(ir, usesAndDefs, null, Env.empty) // Value IR inference doesn't need context
    val smm = InferStreamMemoryManagement(ir, usesAndDefs)
    InferPType(ir, Env.empty, requiredness, usesAndDefs)

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

    val emitContext = new EmitContext(ctx, requiredness, smm)
    val rt = Emit(emitContext, ir, fb, expectedCodeReturnType)

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
  ): (Option[SingleCodeType], (Int, Region) => (F with FunctionWithAggRegion)) = {
    val normalizeNames = new NormalizeNames(_.toString)
    val normalizedBody = normalizeNames(body,
      Env(params.map { case (n, _) => n -> n }: _*))
    val k = CodeCacheKey(aggSigs, params.map { case (n, pt) => (n, pt) }, normalizedBody)
    codeCache.get(k) match {
      case Some(v) =>
        return (v.typ, v.f.asInstanceOf[(Int, Region) => (F with FunctionWithAggRegion)])
      case None =>
    }

    var ir = body
    ir = Subst(ir, BindingEnv(params
      .zipWithIndex
      .foldLeft(Env.empty[IR]) { case (e, ((n, t), i)) => e.bind(n, In(i, t)) }))
    ir = LoweringPipeline.compileLowerer(optimize).apply(ctx, ir).asInstanceOf[IR].noSharing

    TypeCheck(ir, BindingEnv(Env.fromSeq[Type](params.map { case (name, t) => name -> t.virtualType })))

    val usesAndDefs = ComputeUsesAndDefs(ir, errorIfFreeVariables = false)
    val requiredness = Requiredness.apply(ir, usesAndDefs, null, Env.empty) // Value IR inference doesn't need context
    val smm = InferStreamMemoryManagement(ir, usesAndDefs)
    InferPType(ir, Env.empty, requiredness, usesAndDefs)

    val fb = EmitFunctionBuilder[F](ctx, "CompiledWithAggs",
      CodeParamType(typeInfo[Region]) +: params.map { case (_, pt) => pt},
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

    val emitContext = new EmitContext(ctx, requiredness, smm)
    val rt = Emit(emitContext, ir, fb, expectedCodeReturnType, Some(aggSigs))

    val f = fb.resultWithIndex()
    codeCache += k -> CodeCacheValue(rt, f)
    (rt, f.asInstanceOf[(Int, Region) => (F with FunctionWithAggRegion)])
  }
}

object CompileIterator {

  import is.hail.asm4s.joinpoint._

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
  ): (PType, (Int, Region) => F) = {

    val fb = EmitFunctionBuilder.apply[F](ctx, "stream", argTypeInfo.toFastIndexedSeq, CodeParamType(BooleanInfo))
    val outerRegionField = fb.genFieldThisRef[Region]("outerRegion")
    val eltRegionField = fb.genFieldThisRef[Region]("eltRegion")
    val setF = fb.newEmitMethod("setRegions", FastIndexedSeq(CodeParamType(typeInfo[Region]), CodeParamType(typeInfo[Region])), CodeParamType(typeInfo[Unit]))
    setF.emit(Code(outerRegionField := setF.getCodeParam[Region](1), eltRegionField := setF.getCodeParam[Region](2)))

    val stepF = fb.apply_method
    val stepFECB = stepF.ecb

    val outerRegion = StagedRegion(outerRegionField, allowSubregions = false)
    val eltRegion = outerRegion.createChildRegion(stepF)

    val ir = LoweringPipeline.compileLowerer(true)(ctx, body).asInstanceOf[IR].noSharing
    TypeCheck(ir)

    val usesAndDefs = ComputeUsesAndDefs(ir, errorIfFreeVariables = false)
    val requiredness = Requiredness.apply(ir, usesAndDefs, null, Env.empty) // Value IR inference doesn't need context
    val smm = InferStreamMemoryManagement(ir, usesAndDefs)
    InferPType(ir, Env.empty, requiredness, usesAndDefs)

    val emitContext = new EmitContext(ctx, requiredness, smm)
    val emitter = new Emit(emitContext, stepFECB)

    val returnType = ir.pType.asInstanceOf[PStream].elementType.asInstanceOf[PStruct].setRequired(true)

    val optStream = EmitStream.emit(emitter, ir, stepF, outerRegion, Env.empty, None)

    val elementAddress = stepF.genFieldThisRef[Long]("elementAddr")

    val didSetup = stepF.genFieldThisRef[Boolean]("didSetup")
    stepF.cb.emitInit(didSetup := false)

    implicit val ecc: EmitStreamContext = EmitStreamContext(stepF)

    val pullLabel = CodeLabel()
    val eosField = stepF.genFieldThisRef[Boolean]("eos")
    val eosLabel = CodeLabel()

    val source: Source[EmitCode] = optStream.pv.asStream.stream.getStream(eltRegion).apply(
      eosLabel.goto,
      { element =>
        EmitCodeBuilder.scopedVoid(stepF) { cb =>
          val pc = element.toI(cb).get(cb)
          assert(pc.pt.isInstanceOf[PStruct])
          cb.assign(elementAddress, pc.tcode[Long])
          cb += Code._return[Boolean](true)
        }
      })

    Code(eosLabel, source.close, source.close0, eosField := true, Code._return[Boolean](false))

    stepF.emit(
      didSetup.mux(
        eosField.mux(Code._return[Boolean](false), pullLabel.goto),
        Code(
          optStream.setup,
          optStream.m.mux(
            Code._fatal[Unit]("bad stream"),
            Code(
              source.setup0,
              source.setup,
              elementAddress := 0L,
              eosField := false,
              didSetup := true,
              pullLabel,
              source.pull)))))

    val getMB = fb.newEmitMethod("loadAddress", FastIndexedSeq(), LongInfo)
    getMB.emit(elementAddress.load())

    (returnType, fb.resultWithIndex(printWriter))
  }

  def forTableMapPartitions(
    ctx: ExecuteContext,
    typ0: PStruct, typ1: PStream,
    ir: IR
  ): (PType, (Int, RVDContext, Long, StreamArgType) => Iterator[java.lang.Long]) = {
    assert(typ0.required)
    assert(typ1.required)
    val (eltPType, makeStepper) = compileStepper[TMPStepFunction](
      ctx, ir,
      Array[ParamType](
        CodeParamType(typeInfo[Object]),
        SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(typ0)),
        SingleCodeEmitParamType(true, StreamSingleCodeType(typ1.separateRegions, typ1.elementType))),
      None)
    (eltPType, (idx, consumerCtx, v0, part) => {
      val stepper = makeStepper(idx, consumerCtx.partitionRegion)
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
  ): (PType, (Int, RVDContext, Long, Long) => Iterator[java.lang.Long]) = {
    assert(ctxType.required)
    assert(bcValsType.required)
    val (eltPType, makeStepper) = compileStepper[TableStageToRVDStepFunction](
      ctx, ir,
      Array[ParamType](
        CodeParamType(typeInfo[Object]),
        SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(ctxType)),
        SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(bcValsType))),
      None)
    (eltPType, (idx, consumerCtx, v0, v1) => {
      val stepper = makeStepper(idx, consumerCtx.partitionRegion)
      stepper.setRegions(consumerCtx.partitionRegion, consumerCtx.region)
      new LongIteratorWrapper {
        val stepFunction: TableStageToRVDStepFunction = stepper

        def step(): Boolean = stepper.apply(null, v0, v1)
      }
    })
  }

}

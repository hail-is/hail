package is.hail.expr.ir

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend.{ExecuteContext, HailTaskContext}
import is.hail.expr.ir.agg.AggStateSig
import is.hail.expr.ir.lowering.LoweringPipeline
import is.hail.expr.ir.streams.EmitStream
import is.hail.io.fs.FS
import is.hail.rvd.RVDContext
import is.hail.types.physical.{PStruct, PType}
import is.hail.types.physical.stypes.{
  PTypeReferenceSingleCodeType, SingleCodeType, StreamSingleCodeType,
}
import is.hail.types.physical.stypes.interfaces.{NoBoxLongIterator, SStream}
import is.hail.types.virtual.Type
import is.hail.utils._

import java.io.PrintWriter

case class CodeCacheKey(
  aggSigs: IndexedSeq[AggStateSig],
  args: Seq[(String, EmitParamType)],
  body: IR,
)

case class CompiledFunction[T](
  typ: Option[SingleCodeType],
  f: (HailClassLoader, FS, HailTaskContext, Region) => T,
) {
  def tuple: (Option[SingleCodeType], (HailClassLoader, FS, HailTaskContext, Region) => T) =
    (typ, f)
}

object Compile {
  def apply[F: TypeInfo](
    ctx: ExecuteContext,
    params: IndexedSeq[(String, EmitParamType)],
    expectedCodeParamTypes: IndexedSeq[TypeInfo[_]],
    expectedCodeReturnType: TypeInfo[_],
    body: IR,
    optimize: Boolean = true,
    print: Option[PrintWriter] = None,
  ): (Option[SingleCodeType], (HailClassLoader, FS, HailTaskContext, Region) => F) = {

    val normalizedBody =
      new NormalizeNames(_.toString)(ctx, body, Env(params.map { case (n, _) => n -> n }: _*))
    val k =
      CodeCacheKey(FastSeq[AggStateSig](), params.map { case (n, pt) => (n, pt) }, normalizedBody)
    (ctx.backend.lookupOrCompileCachedFunction[F](k) {

      var ir = body
      ir = Subst(
        ir,
        BindingEnv(params
          .zipWithIndex
          .foldLeft(Env.empty[IR]) { case (e, ((n, t), i)) => e.bind(n, In(i, t)) }),
      )
      ir = LoweringPipeline.compileLowerer(optimize).apply(ctx, ir).asInstanceOf[IR].noSharing(ctx)

      TypeCheck(ctx, ir, BindingEnv.empty)

      val returnParam = CodeParamType(SingleCodeType.typeInfoFromType(ir.typ))

      val fb = EmitFunctionBuilder[F](
        ctx,
        "Compiled",
        CodeParamType(typeInfo[Region]) +: params.map { case (_, pt) =>
          pt
        },
        returnParam,
        Some("Emit.scala"),
      )

      /* { def visit(x: IR): Unit = { println(f"${ System.identityHashCode(x) }%08x ${
       * x.getClass.getSimpleName } ${ x.pType }") Children(x).foreach { case c: IR => visit(c) } }
       *
       * visit(ir) } */

      assert(
        fb.mb.parameterTypeInfo == expectedCodeParamTypes,
        s"expected $expectedCodeParamTypes, got ${fb.mb.parameterTypeInfo}",
      )
      assert(
        fb.mb.returnTypeInfo == expectedCodeReturnType,
        s"expected $expectedCodeReturnType, got ${fb.mb.returnTypeInfo}",
      )

      val emitContext = EmitContext.analyze(ctx, ir)
      val rt = Emit(emitContext, ir, fb, expectedCodeReturnType, params.length)
      CompiledFunction(rt, fb.resultWithIndex(print))
    }).tuple
  }
}

object CompileWithAggregators {
  def apply[F: TypeInfo](
    ctx: ExecuteContext,
    aggSigs: Array[AggStateSig],
    params: IndexedSeq[(String, EmitParamType)],
    expectedCodeParamTypes: IndexedSeq[TypeInfo[_]],
    expectedCodeReturnType: TypeInfo[_],
    body: IR,
    optimize: Boolean = true,
  ): (
    Option[SingleCodeType],
    (HailClassLoader, FS, HailTaskContext, Region) => (F with FunctionWithAggRegion),
  ) = {
    val normalizedBody =
      new NormalizeNames(_.toString)(ctx, body, Env(params.map { case (n, _) => n -> n }: _*))
    val k = CodeCacheKey(aggSigs, params.map { case (n, pt) => (n, pt) }, normalizedBody)
    (ctx.backend.lookupOrCompileCachedFunction[F with FunctionWithAggRegion](k) {

      var ir = body
      ir = Subst(
        ir,
        BindingEnv(params
          .zipWithIndex
          .foldLeft(Env.empty[IR]) { case (e, ((n, t), i)) => e.bind(n, In(i, t)) }),
      )
      ir = LoweringPipeline.compileLowerer(optimize).apply(ctx, ir).asInstanceOf[IR].noSharing(ctx)

      TypeCheck(
        ctx,
        ir,
        BindingEnv(Env.fromSeq[Type](params.map { case (name, t) => name -> t.virtualType })),
      )

      val fb = EmitFunctionBuilder[F](
        ctx,
        "CompiledWithAggs",
        CodeParamType(typeInfo[Region]) +: params.map { case (_, pt) => pt },
        SingleCodeType.typeInfoFromType(ir.typ),
        Some("Emit.scala"),
      )

      /* { def visit(x: IR): Unit = { println(f"${ System.identityHashCode(x) }%08x ${
       * x.getClass.getSimpleName } ${ x.pType }") Children(x).foreach { case c: IR => visit(c) } }
       *
       * visit(ir) } */

      val emitContext = EmitContext.analyze(ctx, ir)
      val rt = Emit(emitContext, ir, fb, expectedCodeReturnType, params.length, Some(aggSigs))

      val f = fb.resultWithIndex()
      CompiledFunction(
        rt,
        f.asInstanceOf[(
          HailClassLoader,
          FS,
          HailTaskContext,
          Region,
        ) => (F with FunctionWithAggRegion)],
      )
    }).tuple
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
    def apply(o: Object, a: Long, b: NoBoxLongIterator): Boolean

    def setRegions(outerRegion: Region, eltRegion: Region): Unit
  }

  abstract private class LongIteratorWrapper extends Iterator[java.lang.Long] {
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

  private def compileStepper[F >: Null <: StepFunctionBase: TypeInfo](
    ctx: ExecuteContext,
    body: IR,
    argTypeInfo: Array[ParamType],
    printWriter: Option[PrintWriter],
  ): (PType, (HailClassLoader, FS, HailTaskContext, Region) => F) = {

    val fb = EmitFunctionBuilder.apply[F](
      ctx,
      s"stream_${body.getClass.getSimpleName}",
      argTypeInfo.toFastSeq,
      CodeParamType(BooleanInfo),
      Some("Emit.scala"),
    )
    val outerRegionField = fb.genFieldThisRef[Region]("outerRegion")
    val eltRegionField = fb.genFieldThisRef[Region]("eltRegion")
    val setF = fb.newEmitMethod(
      "setRegions",
      FastSeq(CodeParamType(typeInfo[Region]), CodeParamType(typeInfo[Region])),
      CodeParamType(typeInfo[Unit]),
    )
    setF.emit(Code(
      outerRegionField := setF.getCodeParam[Region](1),
      eltRegionField := setF.getCodeParam[Region](2),
    ))

    val stepF = fb.apply_method
    val stepFECB = stepF.ecb

    val outerRegion = outerRegionField

    val ir = LoweringPipeline.compileLowerer(true)(ctx, body).asInstanceOf[IR].noSharing(ctx)
    TypeCheck(ctx, ir)

    var elementAddress: Settable[Long] = null
    var returnType: PType = null

    stepF.emitWithBuilder[Boolean] { cb =>
      val emitContext = EmitContext.analyze(ctx, ir)
      val emitter = new Emit(emitContext, stepFECB)

      val env = EmitEnv(
        Env.empty,
        argTypeInfo.indices.filter(i => argTypeInfo(i).isInstanceOf[EmitParamType]).map(i =>
          stepF.getEmitParam(cb, i + 1)
        ),
      )
      val optStream = EmitCode.fromI(stepF)(cb =>
        EmitStream.produce(emitter, ir, cb, cb.emb, outerRegion, env, None)
      )
      returnType = optStream.st.asInstanceOf[SStream].elementEmitType.storageType.setRequired(true)

      elementAddress = stepF.genFieldThisRef[Long]("elementAddr")

      val didSetup = stepF.genFieldThisRef[Boolean]("didSetup")
      stepF.cb.emitInit(didSetup := false)

      val eosField = stepF.genFieldThisRef[Boolean]("eos")

      val producer = optStream.pv.asStream.getProducer(cb.emb)

      val ret = cb.newLocal[Boolean]("stepf_ret")
      val Lreturn = CodeLabel()

      cb.if_(
        !didSetup, {
          optStream.toI(cb).get(cb) // handle missing, but bound stream producer above

          cb.assign(producer.elementRegion, eltRegionField)
          producer.initialize(cb, outerRegion)
          cb.assign(didSetup, true)
          cb.assign(eosField, false)
        },
      )

      cb.if_(
        eosField, {
          cb.assign(ret, false)
          cb.goto(Lreturn)
        },
      )

      cb.goto(producer.LproduceElement)

      stepF.implementLabel(producer.LendOfStream) { cb =>
        producer.close(cb)
        cb.assign(eosField, true)
        cb.assign(ret, false)
        cb.goto(Lreturn)
      }

      stepF.implementLabel(producer.LproduceElementDone) { cb =>
        val pc = producer.element.toI(cb).get(cb)
        cb.assign(elementAddress, returnType.store(cb, producer.elementRegion, pc, false))
        cb.assign(ret, true)
        cb.goto(Lreturn)
      }

      cb.define(Lreturn)
      ret
    }

    val getMB = fb.newEmitMethod("loadAddress", FastSeq(), LongInfo)
    getMB.emit(elementAddress.load())

    (returnType, fb.resultWithIndex(printWriter))
  }

  def forTableMapPartitions(
    ctx: ExecuteContext,
    typ0: PStruct,
    streamElementType: PType,
    ir: IR,
  ): (
    PType,
    (HailClassLoader, FS, HailTaskContext, RVDContext, Long, NoBoxLongIterator) => Iterator[java.lang.Long],
  ) = {
    assert(typ0.required)
    assert(streamElementType.required)
    val (eltPType, makeStepper) = compileStepper[TMPStepFunction](
      ctx,
      ir,
      Array[ParamType](
        CodeParamType(typeInfo[Object]),
        SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(typ0)),
        SingleCodeEmitParamType(true, StreamSingleCodeType(true, streamElementType, true)),
      ),
      None,
    )
    (
      eltPType,
      (theHailClassLoader, fs, htc, consumerCtx, v0, part) => {
        val stepper = makeStepper(theHailClassLoader, fs, htc, consumerCtx.partitionRegion)
        stepper.setRegions(consumerCtx.partitionRegion, consumerCtx.region)
        new LongIteratorWrapper {
          val stepFunction: TMPStepFunction = stepper

          def step(): Boolean = stepper.apply(null, v0, part)
        }
      },
    )
  }

  def forTableStageToRVD(
    ctx: ExecuteContext,
    ctxType: PStruct,
    bcValsType: PType,
    ir: IR,
  ): (
    PType,
    (HailClassLoader, FS, HailTaskContext, RVDContext, Long, Long) => Iterator[java.lang.Long],
  ) = {
    assert(ctxType.required)
    assert(bcValsType.required)
    val (eltPType, makeStepper) = compileStepper[TableStageToRVDStepFunction](
      ctx,
      ir,
      Array[ParamType](
        CodeParamType(typeInfo[Object]),
        SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(ctxType)),
        SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(bcValsType)),
      ),
      None,
    )
    (
      eltPType,
      (theHailClassLoader, fs, htc, consumerCtx, v0, v1) => {
        val stepper = makeStepper(theHailClassLoader, fs, htc, consumerCtx.partitionRegion)
        stepper.setRegions(consumerCtx.partitionRegion, consumerCtx.region)
        new LongIteratorWrapper {
          val stepFunction: TableStageToRVDStepFunction = stepper

          def step(): Boolean = stepper.apply(null, v0, v1)
        }
      },
    )
  }

}

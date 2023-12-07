package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s.{AsmFunction1RegionLong, AsmFunction2, LongInfo, Value}
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{
  Compile, Emit, EmitClassBuilder, EmitCode, EmitCodeBuilder, EmitContext, EmitEnv,
  EmitMethodBuilder, Env, IEmitCode, IR, SCodeEmitParamType,
}
import is.hail.types.physical.PType
import is.hail.types.physical.stypes.{EmitType, SType}
import is.hail.types.virtual.Type

// (IR => T), seq op (IR T => T), and comb op (IR (T,T) => T)

class FoldAggregator(
  val resultEmitType: EmitType,
  accumName: String,
  otherAccumName: String,
  combOpIR: IR,
) extends StagedAggregator {
  override type State = TypedRegionBackedAggState

  override val initOpTypes: Seq[Type] = IndexedSeq(resultEmitType.virtualType)

  override val seqOpTypes: Seq[Type] = IndexedSeq(resultEmitType.virtualType)

  override protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    val Array(initEC) = init
    initEC.toI(cb).consume(cb, state.storeMissing(cb), sv => state.storeNonmissing(cb, sv))
  }

  override protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(elt: EmitCode) = seq
    elt.toI(cb).consume(cb, state.storeMissing(cb), sv => state.storeNonmissing(cb, sv))
  }

  override protected def _combOp(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    region: Value[Region],
    state: TypedRegionBackedAggState,
    other: TypedRegionBackedAggState,
  ): Unit = {
    val stateEV = state.get(cb).memoizeField(cb, "fold_agg_comb_op_state")
    val otherEV = other.get(cb).memoizeField(cb, "fold_agg_comb_op_other")
    val env = EmitEnv(Env.apply((accumName, stateEV), (otherAccumName, otherEV)), IndexedSeq())
    val pEnv = Env.apply(
      (accumName, resultEmitType.storageType),
      (otherAccumName, resultEmitType.storageType),
    )

    val emitCtx = EmitContext.analyze(ctx, combOpIR, pEnv)
    val emit = new Emit[Any](emitCtx, cb.emb.ecb.asInstanceOf[EmitClassBuilder[Any]])
    val ec = emit.emit(combOpIR, cb.emb.asInstanceOf[EmitMethodBuilder[Any]], region, env, None)
    ec.toI(cb).consume(cb, state.storeMissing(cb), sv => state.storeNonmissing(cb, sv))
  }

  override protected def _result(cb: EmitCodeBuilder, state: State, region: Value[Region])
    : IEmitCode =
    state.get(cb).map(cb)(sv => sv.copyToRegion(cb, region, sv.st))
}

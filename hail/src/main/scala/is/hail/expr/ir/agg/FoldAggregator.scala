package is.hail.expr.ir.agg
import is.hail.annotations.Region
import is.hail.asm4s.{AsmFunction1RegionLong, AsmFunction2, LongInfo, Value}
import is.hail.expr.ir.{Compile, Emit, EmitClassBuilder, EmitCode, EmitCodeBuilder, EmitContext, EmitEnv, EmitMethodBuilder, Env, ExecuteContext, IR, SCodeEmitParamType}
import is.hail.types.physical.PType
import is.hail.types.physical.stypes.{EmitType, SType}
import is.hail.types.virtual.Type

// (IR => T), seq op (IR T => T), and comb op (IR (T,T) => T)

class FoldAggregator(val initOpTypes: Seq[Type], val seqOpTypes: Seq[Type], val resultType: PType, accumName: String, otherAccumName: String, combOpIR: IR) extends StagedAggregator {
  override type State = TypedRegionBackedAggState

  override protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    val Array(initEC) = init
    initEC.toI(cb).consume(cb, state.storeMissing(cb), sv => state.storeNonmissing(cb, sv.get))
  }

  override protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(elt: EmitCode) = seq
    elt.toI(cb).consume(cb, state.storeMissing(cb), sv => state.storeNonmissing(cb, sv.get))
  }

  override protected def _combOp(ctx: ExecuteContext, cb: EmitCodeBuilder, state: TypedRegionBackedAggState, other: TypedRegionBackedAggState): Unit = {

    val stateEV = state.get(cb).memoize(cb, "fold_agg_comb_op_state")
    val otherEV = other.get(cb).memoize(cb, "fold_agg_comb_op_other")
    val env = EmitEnv(Env.apply((accumName, stateEV), (otherAccumName, otherEV)), IndexedSeq())
    val pEnv = Env.apply((accumName, stateEV.st.storageType()), (otherAccumName, otherEV.st.storageType()))


    // What if stuff is let bound in there from outside of the combop? Invalid, figure out where to catch that (TypeCheck probably).
    val emitCtx = EmitContext.analyze(ctx, combOpIR, pEnv)
    val emit = new Emit[Any](emitCtx, cb.emb.ecb.asInstanceOf[EmitClassBuilder[Any]])
    val ec = emit.emit(combOpIR, cb.emb.asInstanceOf[EmitMethodBuilder[Any]], env, None)
    ec.toI(cb).consume(cb, state.storeMissing(cb), sv => state.storeNonmissing(cb, sv.get))
  }

  override protected def _storeResult(cb: EmitCodeBuilder, state: State, pt: PType, addr: Value[Long], region: Value[Region], ifMissing: EmitCodeBuilder => Unit): Unit = {
    state.get(cb).consume(cb,
      ifMissing(cb),
      { sv => pt.storeAtAddress(cb, addr, region, sv.get, deepCopy = true) })
  }
}

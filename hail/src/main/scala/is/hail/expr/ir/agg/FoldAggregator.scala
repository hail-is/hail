package is.hail.expr.ir.agg
import is.hail.annotations.Region
import is.hail.asm4s.{AsmFunction1RegionLong, AsmFunction2, LongInfo, Value}
import is.hail.expr.ir.{Compile, Emit, EmitClassBuilder, EmitCode, EmitCodeBuilder, EmitContext, EmitEnv, EmitMethodBuilder, Env, ExecuteContext, IR, SCodeEmitParamType}
import is.hail.types.physical.PType
import is.hail.types.physical.stypes.{EmitType, SType}
import is.hail.types.virtual.Type

// (IR => T), seq op (IR T => T), and comb op (IR (T,T) => T)

class FoldAggregator(val initOpTypes: Seq[Type], val seqOpTypes: Seq[Type], val resultType: PType, combOpIR: IR) extends StagedAggregator {
  override type State = TypedRegionBackedAggState

  override protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    val Array(initEC) = init
    initEC.toI(cb).consume(cb, state.storeMissing(cb), sc => state.storeNonmissing(cb, sc))
  }

  override protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(elt: EmitCode) = seq
    elt.toI(cb).consume(cb, state.storeMissing(cb), sc => state.storeNonmissing(cb, sc))
  }

  override protected def _combOp(ctx: ExecuteContext, cb: EmitCodeBuilder, state: TypedRegionBackedAggState, other: TypedRegionBackedAggState): Unit = {
    // TODO: Figure out how to actually compile the CombOpIR. It's going to have to be an IR that takes in parameters I think, how does that work?
    // What if stuff is let bound in there from outside of the combop? Invalid, figure out where to catch that (TypeCheck probably).
    val emitCtx = EmitContext.analyze(ctx, combOpIR)
    val emit = new Emit[Any](emitCtx, cb.emb.ecb.asInstanceOf[EmitClassBuilder[Any]])
    val env = EmitEnv(Env.apply(???, ???), IndexedSeq())
    val ec = emit.emit(combOpIR, cb.emb.asInstanceOf[EmitMethodBuilder[Any]], env, None)
    ec.toI(cb).consume(cb, cb._fatal("Haven't thought through yet"), sc => state.storeNonmissing(cb, sc))
    ???
  }

  override protected def _storeResult(cb: EmitCodeBuilder, state: State, pt: PType, addr: Value[Long], region: Value[Region], ifMissing: EmitCodeBuilder => Unit): Unit = {
    state.get(cb).consume(cb,
      ifMissing(cb),
      { sc => pt.storeAtAddress(cb, addr, region, sc, deepCopy = true) })
  }
}

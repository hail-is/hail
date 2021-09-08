package is.hail.expr.ir.agg
import is.hail.annotations.Region
import is.hail.asm4s.Value
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder}
import is.hail.types.physical.PType
import is.hail.types.virtual.Type

// (IR => T), seq op (IR T => T), and comb op (IR (T,T) => T)

class FoldAggregator(val initOpTypes: Seq[Type], val seqOpTypes: Seq[Type], val resultType: PType) extends StagedAggregator {
  override type State = TypedRegionBackedAggState

  override protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    ???
  }

  override protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = ???

  override protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit = ???

  override protected def _storeResult(cb: EmitCodeBuilder, state: State, pt: PType, addr: Value[Long], region: Value[Region], ifMissing: EmitCodeBuilder => Unit): Unit = ???
}

package is.hail.expr.ir.agg

import is.hail.annotations.StagedRegionValueBuilder
import is.hail.asm4s._
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitCodeBuilder}
import is.hail.expr.types.physical._

object CountAggregator extends StagedAggregator {
  type State = PrimitiveRVAState

  val resultType: PType = PInt64(true)

  def createState(cb: EmitCodeBuilder): State = new PrimitiveRVAState(Array(PInt64(true)), cb.emb.ecb)

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    assert(init.length == 0)
    val (_, v, _) = state.fields(0)
    cb.assignAny(v, 0L)
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    assert(seq.length == 0)
    val (_, v, _) = state.fields(0)
    cb.assignAny(v, coerce[Long](v) + 1L)
  }

  protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit = {
    val (_, v1, _) = state.fields(0)
    val (_, v2, _) = other.fields(0)
    cb.assignAny(v1, coerce[Long](v1) + coerce[Long](v2))
  }

  protected def _result(cb: EmitCodeBuilder, state: State, srvb: StagedRegionValueBuilder): Unit = {
    val (_, v, _) = state.fields(0)
    cb += srvb.addLong(coerce[Long](v))
  }
}

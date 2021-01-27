package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder}
import is.hail.types.physical._
import is.hail.types.virtual.Type

object CountAggregator extends StagedAggregator {
  type State = PrimitiveRVAState

  val resultType: PType = PInt64(true)
  val initOpTypes: Seq[Type] = Array[Type]()
  val seqOpTypes: Seq[Type] = Array[Type]()

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    assert(init.length == 0)
    assert(state.vtypes.head.r.required)
    val (_, v, _) = state.fields(0)
    cb.assignAny(v, 0L)
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    assert(seq.length == 0)
    assert(state.vtypes.head.r.required)
    val (_, v, _) = state.fields(0)
    cb.assignAny(v, coerce[Long](v) + 1L)
  }

  protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit = {
    assert(state.vtypes.head.r.required)
    val (_, v1, _) = state.fields(0)
    val (_, v2, _) = other.fields(0)
    cb.assignAny(v1, coerce[Long](v1) + coerce[Long](v2))
  }

  protected def _storeResult(cb: EmitCodeBuilder, state: State, pt: PType, addr: Value[Long], region: Value[Region], ifMissing: EmitCodeBuilder => Unit): Unit = {
    assert(state.vtypes.head.r.required)
    val (_, v, _) = state.fields(0)
    pt.storeAtAddress(cb, addr, region, PCode(PInt64Required, v), deepCopy = true)
  }
}

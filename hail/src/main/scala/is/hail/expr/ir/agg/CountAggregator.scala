package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.physical._
import is.hail.utils._

object CountAggregator extends StagedRegionValueAggregator {
  type State = PrimitiveRVAState

  val initOpTypes: Array[PType] = Array()
  val seqOpTypes: Array[PType] = Array()
  val resultType: PType = PInt64()

  def createState(mb: EmitMethodBuilder): State = PrimitiveRVAState(Array(PInt64(true)), mb)

  def initOp(state: State, init: Array[RVAVariable], dummy: Boolean): Code[Unit] = {
    assert(init.length == 0)
    val (_, v, _) = state.fields(0)
    Code(v.storeAny(0L), state._loaded := true)
  }

  def seqOp(state: State, seq: Array[RVAVariable], dummy: Boolean): Code[Unit] = {
    assert(seq.length == 0)
    val (_, v, _) = state.fields(0)
    v.storeAny(coerce[Long](v) + 1L)
  }

  def combOp(state: State, other: State, dummy: Boolean): Code[Unit] = {
    val (_, v1, _) = state.fields(0)
    val (_, v2, _) = other.fields(0)
    v1.storeAny(coerce[Long](v1) + coerce[Long](v2))
  }

  def result(state: State, srvb: StagedRegionValueBuilder, dummy: Boolean): Code[Unit] = {
    val (_, v, _) = state.fields(0)
    srvb.addLong(coerce[Long](v))
  }
}


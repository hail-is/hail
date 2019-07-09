package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.physical._
import is.hail.utils._

object CountAggregator extends StagedRegionValueAggregator {
  type State = TypedRVAState

  val initOpTypes: Array[PType] = Array()
  val seqOpTypes: Array[PType] = Array()
  val resultType: PType = PInt64()

  private val stateType: PTuple = PTuple(PInt64(true))

  def createState(mb: EmitMethodBuilder): State = TypedRVAState(PTuple(PInt64(true)), mb, mb.newField[Region], mb.newField[Long])

  def initOp(state: State, init: Array[RVAVariable], dummy: Boolean): Code[Unit] = {
    assert(init.length == 0)
    Code(
      state.off := state.region.allocate(stateType.alignment, stateType.byteSize),
      state.region.storeLong(stateType.fieldOffset(state.off, 0), 0L))
  }

  def seqOp(state: State, seq: Array[RVAVariable], dummy: Boolean): Code[Unit] = {
    assert(seq.length == 0)
    val off = stateType.fieldOffset(state.off, 0)
    state.region.storeLong(off, state.region.loadLong(off) + 1L)
  }

  def combOp(state: State, other: State, dummy: Boolean): Code[Unit] = {
    val off1 = stateType.fieldOffset(state.off, 0)
    val off2 = stateType.fieldOffset(other.off, 0)
    state.region.storeLong(off1, state.region.loadLong(off1) + state.region.loadLong(off2))
  }

  def result(state: State, srvb: StagedRegionValueBuilder, dummy: Boolean): Code[Unit] =
    srvb.addLong(state.region.loadLong(stateType.fieldOffset(state.off, 0)))
}


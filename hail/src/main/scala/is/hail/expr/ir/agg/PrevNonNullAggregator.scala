package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.physical._
import is.hail.utils._

class PrevNonNullAggregator(typ: PType) extends StagedRegionValueAggregator {
  type State = TypedRVAState

  val initOpTypes: Array[PType] = Array()
  val seqOpTypes: Array[PType] = Array(typ)

  val stateType: PTuple = PTuple(FastIndexedSeq(typ))
  val resultType: PType = typ

  def createState(mb: EmitMethodBuilder): State = TypedRVAState(PTuple(FastIndexedSeq(typ)), mb, mb.newField[Region], mb.newField[Long])

  def initOp(state: State, init: Array[RVAVariable], dummy: Boolean): Code[Unit] = {
    assert(init.length == 0)
    Code(
      state.off := state.region.allocate(stateType.alignment, stateType.byteSize),
      stateType.setFieldMissing(state.region, state.off, 0))
  }

  def seqOp(state: State, seq: Array[RVAVariable], dummy: Boolean): Code[Unit] = {
    val Array(elt: RVAVariable) = seq
    assert(elt.t == typ)

    val copyValue = typ match {
      case _: PBoolean => state.region.storeByte(stateType.fieldOffset(state.off, 0), elt.v[Boolean].toI.toB)
      case _: PInt32 => state.region.storeInt(stateType.fieldOffset(state.off, 0), elt.v[Int])
      case _: PInt64 => state.region.storeLong(stateType.fieldOffset(state.off, 0), elt.v[Long])
      case _: PFloat32 => state.region.storeFloat(stateType.fieldOffset(state.off, 0), elt.v[Float])
      case _: PFloat64 => state.region.storeDouble(stateType.fieldOffset(state.off, 0), elt.v[Double])
      case _ =>
        val v = state.mb.newField[Long]
        Code(v := elt.v[Long],
          StagedRegionValueBuilder.deepCopy(state.er, elt.t, v, stateType.fieldOffset(state.off, 0)))
    }

    elt.m.mux(Code._empty, Code(
      state.region.refreshRegion(),
      state.off := state.region.allocate(stateType.alignment, stateType.byteSize),
      stateType.clearMissingBits(state.region, state.off),
      elt.setup,
      copyValue))
  }

  def combOp(state: State, other: State, dummy: Boolean): Code[Unit] = {
    stateType.isFieldMissing(other.region, other.off, 0).mux(
      Code._empty,
      Code(
        state.region.refreshRegion(),
        state.off := StagedRegionValueBuilder.deepCopy(other.er, stateType, other.off)))
  }

  def result(state: State, srvb: StagedRegionValueBuilder, dummy: Boolean): Code[Unit] = {
    stateType.isFieldMissing(state.region, state.off, 0).mux(
      srvb.setMissing(),
      srvb.addWithDeepCopy(resultType, stateType.loadField(state.region, state.off, 0)))
  }
}

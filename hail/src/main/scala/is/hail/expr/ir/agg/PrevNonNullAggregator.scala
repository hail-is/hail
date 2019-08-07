package is.hail.expr.ir.agg

import is.hail.annotations.StagedRegionValueBuilder
import is.hail.asm4s._
import is.hail.expr.ir.{EmitFunctionBuilder, EmitTriplet}
import is.hail.expr.types.physical._
import is.hail.utils._

class PrevNonNullAggregator(typ: PType) extends StagedAggregator {
  type State = TypedRVAState

  val stateType: PTuple = PTuple(typ.setRequired(false))
  val resultType: PType = typ

  def createState(fb: EmitFunctionBuilder[_]): State =
    new TypedRVAState(stateType, fb)

  def initOp(state: State, init: Array[EmitTriplet], dummy: Boolean): Code[Unit] = {
    assert(init.length == 0)
    Code(
      state.off := state.region.allocate(stateType.alignment, stateType.byteSize),
      stateType.setFieldMissing(state.region, state.off, 0))
  }

  def seqOp(state: State, seq: Array[EmitTriplet], dummy: Boolean): Code[Unit] = {
    val Array(elt: EmitTriplet) = seq

    val copyValue = typ match {
      case _: PBoolean => state.region.storeByte(stateType.fieldOffset(state.off, 0), elt.value[Boolean].toI.toB)
      case _: PInt32 => state.region.storeInt(stateType.fieldOffset(state.off, 0), elt.value[Int])
      case _: PInt64 => state.region.storeLong(stateType.fieldOffset(state.off, 0), elt.value[Long])
      case _: PFloat32 => state.region.storeFloat(stateType.fieldOffset(state.off, 0), elt.value[Float])
      case _: PFloat64 => state.region.storeDouble(stateType.fieldOffset(state.off, 0), elt.value[Double])
      case _ =>
        val v = state.fb.newField[Long]
        Code(v := elt.value[Long],
          StagedRegionValueBuilder.deepCopy(state.fb, state.region, typ, v, stateType.fieldOffset(state.off, 0)))
    }

    Code(
      elt.setup,
      elt.m.mux(
        Code._empty,
        Code(
          state.newState,
          state.off := state.region.allocate(stateType.alignment, stateType.byteSize),
          stateType.clearMissingBits(state.region, state.off),
          copyValue))
    )
  }

  def combOp(state: State, other: State, dummy: Boolean): Code[Unit] = {
    stateType.isFieldMissing(other.off, 0).mux(
      Code._empty,
      Code(
        state.newState,
        state.off := StagedRegionValueBuilder.deepCopy(state.fb, state.region, stateType, other.off)))
  }

  def result(state: State, srvb: StagedRegionValueBuilder, dummy: Boolean): Code[Unit] = {
    stateType.isFieldMissing(state.region, state.off, 0).mux(
      srvb.setMissing(),
      srvb.addWithDeepCopy(resultType, stateType.loadField(state.region, state.off, 0)))
  }
}

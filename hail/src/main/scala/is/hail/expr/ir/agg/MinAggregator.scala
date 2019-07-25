package is.hail.expr.ir.agg

import is.hail.annotations.StagedRegionValueBuilder
import is.hail.asm4s._
import is.hail.expr.ir.{EmitMethodBuilder, EmitTriplet}
import is.hail.expr.types.physical.{PFloat64, PInt64, PType}

import scala.language.existentials

class MinAggregator(typ: PType) extends StagedAggregator {
  type State = PrimitiveRVAState
  val resultType: PType = typ

  def min(v1: Code[_], v2: Code[_]): Code[_] = typ match {
    case _: PInt64 => (coerce[Long](v1) < coerce[Long](v2)).mux(v1, v2)
    case _: PFloat64 => (coerce[Double](v1) < coerce[Double](v2)).mux(v1, v2)
    case _ => throw new UnsupportedOperationException(s"can't min over type $typ")
  }

  def createState(mb: EmitMethodBuilder): State =
    PrimitiveRVAState(Array(typ.setRequired(false)), mb)

  def initOp(state: State, init: Array[EmitTriplet], dummy: Boolean): Code[Unit] = {
    assert(init.length == 0)
    val (Some(m), v, _) = state.fields(0)
    Code(m.store(true), state._loaded := true)
  }

  def seqOp(state: State, seq: Array[EmitTriplet], dummy: Boolean): Code[Unit] = {
    val Array(elt) = seq
    val (Some(m), v, _) = state.fields(0)
    Code(
      elt.setup,
      m.mux(
        // copy first element if the state is empty
        Code(m.store(elt.m), v.store(elt.value)),
        elt.m.mux(Code._empty,
          // apply min() if neither are missing
          v.storeAny(min(v, elt.value)))))
  }

  def combOp(state: State, other: State, dummy: Boolean): Code[Unit] = {
    val (Some(m1), v1, _) = state.fields(0)
    val (Some(m2), v2, _) = other.fields(0)
    m1.mux(
      Code(m1.store(m2), v1.storeAny(v2)),
      m2.mux(Code._empty,
        v1.storeAny(min(v1, v2))))
  }

  def result(state: State, srvb: StagedRegionValueBuilder, dummy: Boolean): Code[Unit] = {
    val (Some(m), v, _) = state.fields(0)
    m.mux(
      srvb.setMissing(),
      srvb.addIRIntermediate(typ)(v))
  }
}

package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.physical.{PFloat64, PInt64, PTuple, PType}
import is.hail.utils._

class SumAggregator(typ: PType) extends StagedRegionValueAggregator {
  type State = PrimitiveRVAState

  val initOpTypes: Array[PType] = Array()
  val seqOpTypes: Array[PType] = Array(typ)
  val resultType: PType = typ

  def add(v1: Code[_], v2: Code[_]): Code[_] = typ match {
    case _: PInt64 => coerce[Long](v1) + coerce[Long](v2)
    case _: PFloat64 => coerce[Double](v1) + coerce[Double](v2)
    case _ => throw new UnsupportedOperationException(s"can't sum over type $typ")
  }

  def zero: Code[_] = typ match {
    case _: PInt64 => const(0L)
    case _: PFloat64 => const(0.0d)
    case _ => throw new UnsupportedOperationException(s"can't sum over type $typ")
  }

  def createState(mb: EmitMethodBuilder): State = PrimitiveRVAState(Array(typ.setRequired(true)), mb)

  def initOp(state: State, init: Array[RVAVariable], dummy: Boolean): Code[Unit] = {
    assert(init.length == 0)
    val (_, v, _) = state.fields(0)
    Code(v.storeAny(zero), state._loaded := true)
  }

  def seqOp(state: State, seq: Array[RVAVariable], dummy: Boolean): Code[Unit] = {
    val Array(RVAVariable(elt, t)) = seq
    assert(t isOfType typ, s"$t vs $typ")
    val (_, v, _) = state.fields(0)
    Code(
      elt.setup,
      elt.m.mux(Code._empty,
        v.storeAny(add(v, elt.v))))
  }

  def combOp(state: State, other: State, dummy: Boolean): Code[Unit] = {
    val (_, v1, _) = state.fields(0)
    val (_, v2, _) = other.fields(0)
    v1.storeAny(add(v1, v2))
  }

  def result(state: State, srvb: StagedRegionValueBuilder, dummy: Boolean): Code[Unit] = {
    val (_, v, _) = state.fields(0)
    srvb.addIRIntermediate(typ)(v)
  }
}



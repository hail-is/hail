package is.hail.expr.ir.agg

import is.hail.annotations.StagedRegionValueBuilder
import is.hail.asm4s._
import is.hail.expr.ir.{EmitMethodBuilder, EmitTriplet}
import is.hail.expr.types.physical.PType

case class RVAVariable(triplet: EmitTriplet, t: PType) {
  def setup: Code[Unit] = triplet.setup
  def m: Code[Boolean] = triplet.m
  def v[T]: Code[T] = triplet.value[T]
}

abstract class StagedRegionValueAggregator {
  type State <: RVAState
  def eltArgIdx: Int = 0

  def initOpTypes: Array[PType]
  def seqOpTypes: Array[PType]

  def resultType: PType

  def createState(mb: EmitMethodBuilder): State

  def initOp(state: State, init: Array[RVAVariable], dummy: Boolean): Code[Unit]

  def seqOp(state: State, seq: Array[RVAVariable], dummy: Boolean): Code[Unit]

  def combOp(state: State, other: State, dummy: Boolean): Code[Unit]

  def result(state: State, srvb: StagedRegionValueBuilder, dummy: Boolean): Code[Unit]

  def initOp(state: RVAState, init: Array[RVAVariable]): Code[Unit] = initOp(state.asInstanceOf[State], init, dummy = true)
  def seqOp(state: RVAState, seq: Array[RVAVariable]): Code[Unit] = seqOp(state.asInstanceOf[State], seq, dummy = true)
  def combOp(state: RVAState, other: RVAState): Code[Unit] = combOp(state.asInstanceOf[State], other.asInstanceOf[State], dummy = true)
  def result(state: RVAState, srvb: StagedRegionValueBuilder): Code[Unit] = result(state.asInstanceOf[State], srvb, dummy = true)
}

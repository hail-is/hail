package is.hail.expr.ir.agg

import is.hail.annotations.StagedRegionValueBuilder
import is.hail.asm4s._
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitCodeBuilder}
import is.hail.expr.types.physical.PType

abstract class StagedAggregator {
  type State <: AggregatorState

  def resultType: PType

  def createState(cb: EmitCodeBuilder): State

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode])

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode])

  protected def _combOp(cb: EmitCodeBuilder, state: State, other: State)

  protected def _result(cb: EmitCodeBuilder, state: State, srvb: StagedRegionValueBuilder)

  def initOp(cb: EmitCodeBuilder, state: AggregatorState, init: Array[EmitCode]) = _initOp(cb, state.asInstanceOf[State], init)
  def seqOp(cb: EmitCodeBuilder, state: AggregatorState, seq: Array[EmitCode]) = _seqOp(cb, state.asInstanceOf[State], seq)
  def combOp(cb: EmitCodeBuilder, state: AggregatorState, other: AggregatorState) = _combOp(cb, state.asInstanceOf[State], other.asInstanceOf[State])
  def result(cb: EmitCodeBuilder, state: AggregatorState, srvb: StagedRegionValueBuilder) = _result(cb, state.asInstanceOf[State], srvb)
}

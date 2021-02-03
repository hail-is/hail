package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder}
import is.hail.types.physical.PType
import is.hail.types.virtual.Type

abstract class StagedAggregator {
  type State <: AggregatorState

  def resultType: PType
  def initOpTypes: Seq[Type]
  def seqOpTypes: Seq[Type]

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode])

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode])

  protected def _combOp(cb: EmitCodeBuilder, state: State, other: State)

  protected def _storeResult(cb: EmitCodeBuilder, state: State, pt: PType, addr: Value[Long], region: Value[Region], ifMissing: EmitCodeBuilder => Unit): Unit

  def initOp(cb: EmitCodeBuilder, state: AggregatorState, init: Array[EmitCode]) = _initOp(cb, state.asInstanceOf[State], init)
  def seqOp(cb: EmitCodeBuilder, state: AggregatorState, seq: Array[EmitCode]) = _seqOp(cb, state.asInstanceOf[State], seq)
  def combOp(cb: EmitCodeBuilder, state: AggregatorState, other: AggregatorState) = _combOp(cb, state.asInstanceOf[State], other.asInstanceOf[State])
  def storeResult(cb: EmitCodeBuilder, state: AggregatorState, pt: PType, addr: Value[Long], region: Value[Region], ifMissing: EmitCodeBuilder => Unit): Unit =
    _storeResult(cb, state.asInstanceOf[State], pt, addr, region, ifMissing)
}

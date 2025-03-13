package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, IEmitCode}
import is.hail.types.physical.stypes.EmitType
import is.hail.types.virtual.Type

abstract class StagedAggregator {
  type State <: AggregatorState

  def resultEmitType: EmitType
  def initOpTypes: Seq[Type]
  def seqOpTypes: Seq[Type]

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit

  protected def _combOp(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    region: Value[Region],
    state: State,
    other: State,
  ): Unit

  protected def _result(cb: EmitCodeBuilder, state: State, region: Value[Region]): IEmitCode

  def initOp(cb: EmitCodeBuilder, state: AggregatorState, init: Array[EmitCode]) =
    _initOp(cb, state.asInstanceOf[State], init)

  def seqOp(cb: EmitCodeBuilder, state: AggregatorState, seq: Array[EmitCode]) =
    _seqOp(cb, state.asInstanceOf[State], seq)

  def combOp(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    region: Value[Region],
    state: AggregatorState,
    other: AggregatorState,
  ): Unit =
    _combOp(ctx, cb, region, state.asInstanceOf[State], other.asInstanceOf[State])

  def result(cb: EmitCodeBuilder, state: AggregatorState, region: Value[Region]): IEmitCode =
    _result(cb, state.asInstanceOf[State], region)
}

package is.hail.expr.ir.agg

import is.hail.annotations.StagedRegionValueBuilder
import is.hail.asm4s._
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitFunctionBuilder}
import is.hail.expr.types.physical.PType

abstract class StagedAggregator {
  type State <: AggregatorState

  def resultType: PType

  def createState(cb: EmitClassBuilder[_]): State

  protected def _initOp(state: State, init: Array[EmitCode]): Code[Unit]

  protected def _seqOp(state: State, seq: Array[EmitCode]): Code[Unit]

  protected def _combOp(state: State, other: State): Code[Unit]

  protected def _result(state: State, srvb: StagedRegionValueBuilder): Code[Unit]

  def initOp(state: AggregatorState, init: Array[EmitCode]): Code[Unit] = _initOp(state.asInstanceOf[State], init)
  def seqOp(state: AggregatorState, seq: Array[EmitCode]): Code[Unit] = _seqOp(state.asInstanceOf[State], seq)
  def combOp(state: AggregatorState, other: AggregatorState): Code[Unit] = _combOp(state.asInstanceOf[State], other.asInstanceOf[State])
  def result(state: AggregatorState, srvb: StagedRegionValueBuilder): Code[Unit] = _result(state.asInstanceOf[State], srvb)
}

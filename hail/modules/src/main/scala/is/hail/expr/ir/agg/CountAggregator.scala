package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, IEmitCode}
import is.hail.types.physical.stypes.EmitType
import is.hail.types.physical.stypes.interfaces.primitive
import is.hail.types.physical.stypes.primitives.SInt64
import is.hail.types.virtual.Type

object CountAggregator extends StagedAggregator {
  type State = PrimitiveRVAState

  val resultEmitType: EmitType = EmitType(SInt64, true)
  val initOpTypes: Seq[Type] = Array[Type]()
  val seqOpTypes: Seq[Type] = Array[Type]()

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    assert(init.length == 0)
    assert(state.vtypes.head.r.required)
    val ev = state.fields(0)
    cb.assign(ev, EmitCode.present(cb.emb, primitive(const(0L))))
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    assert(seq.length == 0)
    assert(state.vtypes.head.r.required)
    val ev = state.fields(0)
    cb.assign(ev, EmitCode.present(cb.emb, primitive(cb.memoize(ev.pv.asInt64.value + 1L))))
  }

  protected def _combOp(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    region: Value[Region],
    state: PrimitiveRVAState,
    other: PrimitiveRVAState,
  ): Unit = {
    assert(state.vtypes.head.r.required)
    val v1 = state.fields(0)
    val v2 = other.fields(0)
    cb.assign(
      v1,
      EmitCode.present(cb.emb, primitive(cb.memoize(v1.pv.asInt64.value + v2.pv.asInt64.value))),
    )
  }

  protected def _result(cb: EmitCodeBuilder, state: State, region: Value[Region]): IEmitCode = {
    assert(state.vtypes.head.r.required)
    val ev = state.fields(0)
    ev.toI(cb).map(cb)(sv => sv.copyToRegion(cb, region, sv.st))
  }
}

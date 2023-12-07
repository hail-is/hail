package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, IEmitCode}
import is.hail.types.VirtualTypeWithReq
import is.hail.types.physical._
import is.hail.types.physical.stypes.EmitType
import is.hail.types.virtual.Type

class PrevNonNullAggregator(typ: VirtualTypeWithReq) extends StagedAggregator {
  type State = TypedRegionBackedAggState
  val resultEmitType = EmitType(typ.canonicalEmitType.st, false)
  val initOpTypes: Seq[Type] = Array[Type]()
  val seqOpTypes: Seq[Type] = Array[Type](typ.t)

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    assert(init.length == 0)
    state.storeMissing(cb)
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(elt: EmitCode) = seq
    elt.toI(cb)
      .consume(
        cb,
        { /* do nothing if missing */ },
        sc => state.storeNonmissing(cb, sc),
      )
  }

  protected def _combOp(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    region: Value[Region],
    state: TypedRegionBackedAggState,
    other: TypedRegionBackedAggState,
  ): Unit =
    other.get(cb)
      .consume(
        cb,
        { /* do nothing if missing */ },
        sc => state.storeNonmissing(cb, sc),
      )

  protected def _result(cb: EmitCodeBuilder, state: State, region: Value[Region]): IEmitCode =
    state.get(cb).map(cb)(sv => sv.copyToRegion(cb, region, sv.st))
}

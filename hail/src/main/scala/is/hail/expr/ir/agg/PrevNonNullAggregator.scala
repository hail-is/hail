package is.hail.expr.ir.agg

import is.hail.annotations.{Region, RegionUtils, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitCodeBuilder}
import is.hail.types.VirtualTypeWithReq
import is.hail.types.physical._
import is.hail.types.virtual.Type
import is.hail.utils._

class PrevNonNullAggregator(typ: VirtualTypeWithReq) extends StagedAggregator {
  type State = TypedRegionBackedAggState
  private val pt = typ.canonicalPType.setRequired(false)
  val resultType: PType = pt
  val initOpTypes: Seq[Type] = Array[Type]()
  val seqOpTypes: Seq[Type] = Array[Type](typ.t)

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    assert(init.length == 0)
    state.storeMissing(cb)
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(elt: EmitCode) = seq
    elt.toI(cb)
      .consume(cb,
        { /* do nothing if missing */ },
        sc => state.storeNonmissing(cb, sc)
      )
  }

  protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit = {
    other.get()
      .toI(cb)
      .consume(cb,
        { /* do nothing if missing */ },
        sc => state.storeNonmissing(cb, sc)
      )
  }

  protected def _result(cb: EmitCodeBuilder, state: State, srvb: StagedRegionValueBuilder): Unit = {
    val t = state.get()
    cb += Code(
      t.setup,
      t.m.mux(
        srvb.setMissing(),
        srvb.addWithDeepCopy(resultType, t.v)))
  }
}

package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder}
import is.hail.types.VirtualTypeWithReq
import is.hail.types.physical._
import is.hail.types.virtual.Type

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

  protected def _storeResult(cb: EmitCodeBuilder, state: State, pt: PType, addr: Value[Long], region: Value[Region], ifMissing: EmitCodeBuilder => Unit): Unit = {
    state.get().toI(cb).consume(cb,
      ifMissing(cb),
      { sc => pt.storeAtAddress(cb, addr, region, sc, deepCopy = true) })
  }
}

package is.hail.expr.ir.agg

import is.hail.annotations.{Region, RegionUtils, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitCodeBuilder}
import is.hail.types.physical._
import is.hail.utils._

class PrevNonNullAggregator(typ: PType) extends StagedAggregator {
  type State = TypedRegionBackedAggState
  assert(PType.canonical(typ) == typ)
  val resultType: PType = typ
  val initOpTypes: Seq[PType] = Array[PType]()
  val seqOpTypes: Seq[PType] = Array[PType](typ)

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    assert(init.length == 0)
    cb += state.storeMissing()
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(elt: EmitCode) = seq
    val v = state.kb.genFieldThisRef()(typeToTypeInfo(typ))
    cb += Code(
      elt.setup,
      elt.m.mux(Code._empty,
        Code(v := elt.value, state.storeNonmissing(v)))
    )
  }

  protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit = {
    val t = other.get()
    val v = state.kb.genFieldThisRef()(typeToTypeInfo(typ))
    cb += Code(
      t.setup,
      t.m.mux(Code._empty,
        Code(v := t.value, state.storeNonmissing(v))))
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

package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.physical.{PFloat64, PInt64, PTuple, PType}
import is.hail.utils._

class SumAggregator(typ: PType) extends StagedRegionValueAggregator {
  type State = TypedRVAState

  val initOpTypes: Array[PType] = Array()
  val seqOpTypes: Array[PType] = Array(typ)
  val resultType: PType = typ

  private val stateType: PTuple = PTuple(typ.setRequired(true))
  def floatOrLong[T](float: T, long: T): T = typ match {
    case _: PInt64 => long
    case _: PFloat64 => float
    case _ => throw new UnsupportedOperationException(s"can't sum over type $typ")
  }

  def store(r: Code[Region], off: Code[Long], v: Code[_]): Code[Unit] =
    floatOrLong(r.storeDouble(off, coerce[Double](v)), r.storeLong(off, coerce[Long](v)))

  def load[T](r: Code[Region], off: Code[Long]): Code[T] = coerce[T](r.loadIRIntermediate(typ)(off))

  def add(r: Code[Region], off: Code[Long], v: Code[_]): Code[_] =
    floatOrLong(
      r.storeDouble(off, coerce[Double](v)),
      r.storeLong(off, coerce[Long](v)))

  def zero: Code[_] = floatOrLong(const(0.0d), const(0L))

  def createState(mb: EmitMethodBuilder): State = TypedRVAState(PTuple(PInt64(true)), mb, mb.newField[Region], mb.newField[Long])

  def initOp(state: State, init: Array[RVAVariable], dummy: Boolean): Code[Unit] = {
    assert(init.length == 0)
    Code(
      state.off := state.region.allocate(stateType.alignment, stateType.byteSize),
      store(state.region, stateType.fieldOffset(state.off, 0), zero))
  }

  def seqOp(state: State, seq: Array[RVAVariable], dummy: Boolean): Code[Unit] = {
    val Array(RVAVariable(elt, t)) = seq
    assert(t isOfType typ, s"$t vs $typ")
    val off = stateType.fieldOffset(state.off, 0)
    Code(
      elt.setup,
      elt.m.mux(
        Code._empty,
        store(state.region, off,
          floatOrLong(
            load[Double](state.region, off) + elt.value[Double],
            load[Long](state.region, off) + elt.value[Long]))))
  }

  def combOp(state: State, other: State, dummy: Boolean): Code[Unit] = {
    val off1 = stateType.fieldOffset(state.off, 0)
    val off2 = stateType.fieldOffset(other.off, 0)
    store(state.region, off1,
      floatOrLong(
        load[Double](state.region, off1) + load[Double](state.region, off2),
        load[Long](state.region, off1) + load[Long](state.region, off2)))
  }

  def result(state: State, srvb: StagedRegionValueBuilder, dummy: Boolean): Code[Unit] =
    srvb.addIRIntermediate(typ)(load(state.region, stateType.fieldOffset(state.off, 0)))
}



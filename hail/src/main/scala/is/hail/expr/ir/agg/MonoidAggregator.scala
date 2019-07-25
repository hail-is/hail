package is.hail.expr.ir.agg

import is.hail.annotations.StagedRegionValueBuilder
import is.hail.asm4s._
import is.hail.expr.ir.{EmitMethodBuilder, EmitTriplet}
import is.hail.expr.types.physical.{PFloat64, PInt64, PType}

import scala.language.existentials

trait StagedMonoidSpec {
  val typ: PType
  def neutral: Option[Code[_]]
  def apply(v1: Code[_], v2: Code[_]): Code[_]
}

class MonoidAggregator(monoid: StagedMonoidSpec) extends StagedAggregator {
  type State = PrimitiveRVAState
  val typ: PType = monoid.typ
  val resultType: PType = typ

  def createState(mb: EmitMethodBuilder): State =
    PrimitiveRVAState(Array(typ.setRequired(monoid.neutral.isDefined)), mb)

  def initOp(state: State, init: Array[EmitTriplet], dummy: Boolean): Code[Unit] = {
    assert(init.length == 0)
    val (mOpt, v, _) = state.fields(0)
    (mOpt, monoid.neutral) match {
      // if state is optional, then set it to "missing" on initialization
      case (Some(m), _)  => Code(m.store(true), state._loaded := true)
      // if state is not optional, set it to the neutral value
      case (_, Some(v0)) => Code(v.storeAny(v0), state._loaded := true)
    }
  }

  def seqOp(state: State, seq: Array[EmitTriplet], dummy: Boolean): Code[Unit] = {
    val Array(elt) = seq
    val (mOpt, v, _) = state.fields(0)
    Code(elt.setup, combine(mOpt, v, Some(elt.m), elt.v))
  }

  def combOp(state: State, other: State, dummy: Boolean): Code[Unit] = {
    val (m1, v1, _) = state.fields(0)
    val (m2, v2, _) = other.fields(0)
    combine(m1, v1, m2.map(implicitly(_)), v2)
  }

  def result(state: State, srvb: StagedRegionValueBuilder, dummy: Boolean): Code[Unit] = {
    val (mOpt, v, _) = state.fields(0)
    mOpt match {
      case None => srvb.addIRIntermediate(typ)(v)
      case Some(m) =>
        m.mux(
          srvb.setMissing(),
          srvb.addIRIntermediate(typ)(v))
    }
  }

  private def combine(
    m1Opt: Option[ClassFieldRef[Boolean]],
    v1: ClassFieldRef[_],
    m2Opt: Option[Code[Boolean]],
    v2: Code[_]
  ): Code[Unit] = {
    val combineAndStore = v1.storeAny(monoid(v1, v2))
    (m1Opt, m2Opt) match {
      case (None, None) =>
        combineAndStore
      case (None, Some(m2)) =>
        // only update if the element is not missing
        m2.mux(Code._empty, combineAndStore)
      case (Some(m1), None) =>
        m1.mux(
          Code(m1.store(false), v1.storeAny(v2)),
          combineAndStore)
      case (Some(m1), Some(m2)) =>
        m1.mux(
          // if the current state is missing, then just copy the other
          // element + its missingness
          Code(m1.store(m2), v1.storeAny(v2)),
          m2.mux(Code._empty, combineAndStore))
    }
  }
}

class MinMonoid(val typ: PType) extends StagedMonoidSpec {

  def neutral: Option[Code[_]] = None

  def apply(v1: Code[_], v2: Code[_]): Code[_] = typ match {
    case _: PInt64 => (coerce[Long](v1) < coerce[Long](v2)).mux(v1, v2)
    case _: PFloat64 => (coerce[Double](v1) < coerce[Double](v2)).mux(v1, v2)
    case _ => throw new UnsupportedOperationException(s"can't min over type $typ")
  }
}

class SumMonoid(val typ: PType) extends StagedMonoidSpec {

  def neutral: Option[Code[_]] = Some(typ match {
    case _: PInt64 => const(0L)
    case _: PFloat64 => const(0.0d)
    case _ => throw new UnsupportedOperationException(s"can't sum over type $typ")
  })

  def apply(v1: Code[_], v2: Code[_]): Code[_] = typ match {
    case _: PInt64 => coerce[Long](v1) + coerce[Long](v2)
    case _: PFloat64 => coerce[Double](v1) + coerce[Double](v2)
    case _ => throw new UnsupportedOperationException(s"can't sum over type $typ")
  }
}

class MinAggregator(typ: PType) extends MonoidAggregator(new MinMonoid(typ))
class SumAggregator(typ: PType) extends MonoidAggregator(new SumMonoid(typ))

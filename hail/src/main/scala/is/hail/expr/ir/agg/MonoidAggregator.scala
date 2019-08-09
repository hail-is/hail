package is.hail.expr.ir.agg

import is.hail.annotations.StagedRegionValueBuilder
import is.hail.asm4s._
import is.hail.expr.ir.{coerce => _, _}
import is.hail.expr.ir.functions.UtilFunctions
import is.hail.expr.types.physical.{PInt32, PInt64, PFloat32, PFloat64, PType}

import scala.language.existentials
import scala.reflect.ClassTag

trait StagedMonoidSpec {
  val typ: PType
  def neutral: Option[Code[_]]
  def apply(v1: Code[_], v2: Code[_]): Code[_]
}

class MonoidAggregator(monoid: StagedMonoidSpec) extends StagedAggregator {
  type State = PrimitiveRVAState
  val typ: PType = monoid.typ
  val resultType: PType = typ

  def createState(fb: EmitFunctionBuilder[_]): State =
    new PrimitiveRVAState(Array(typ.setRequired(monoid.neutral.isDefined)), fb)

  def initOp(state: State, init: Array[EmitTriplet], dummy: Boolean): Code[Unit] = {
    assert(init.length == 0)
    val (mOpt, v, _) = state.fields(0)
    (mOpt, monoid.neutral) match {
      case (Some(m), _)  => m.store(true)
      case (_, Some(v0)) => v.storeAny(v0)
    }
  }

  def seqOp(state: State, seq: Array[EmitTriplet], dummy: Boolean): Code[Unit] = {
    val Array(elt) = seq
    val (mOpt, v, _) = state.fields(0)
    val eltm = state.fb.newField[Boolean]
    val eltv = state.fb.newField(typeToTypeInfo(typ))
    Code(elt.setup,
      eltm := elt.m,
      eltm.mux(Code._empty, eltv := elt.value),
      combine(mOpt, v, Some(eltm), eltv)
    )
  }

  def combOp(state: State, other: State, dummy: Boolean): Code[Unit] = {
    val (m1, v1, _) = state.fields(0)
    val (m2, v2, _) = other.fields(0)
    combine(m1, v1, m2.map(_.load), v2.load)
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

class ComparisonMonoid(val typ: PType, val functionName: String) extends StagedMonoidSpec {

  def neutral: Option[Code[_]] = None

  private def cmp[T](v1: Code[T], v2: Code[T])(implicit tct: ClassTag[T]): Code[T] =
    Code.invokeStatic[Math,T,T,T](functionName, v1, v2)

  private def nancmp[T](v1: Code[T], v2: Code[T])(implicit tct: ClassTag[T]): Code[T] =
    Code.invokeScalaObject[T,T,T](UtilFunctions.getClass, "nan" + functionName, v1, v2)

  def apply(v1: Code[_], v2: Code[_]): Code[_] = typ match {
    case _: PInt32 => cmp[Int](coerce(v1), coerce(v2))
    case _: PInt64 => cmp[Long](coerce(v1), coerce(v2))
    case _: PFloat32 => nancmp[Float](coerce(v1), coerce(v2))
    case _: PFloat64 => nancmp[Double](coerce(v1), coerce(v2))
    case _ => throw new UnsupportedOperationException(s"can't $functionName over type $typ")
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

class ProductMonoid(val typ: PType) extends StagedMonoidSpec {

  def neutral: Option[Code[_]] = Some(typ match {
    case _: PInt64 => const(1L)
    case _: PFloat64 => const(1.0d)
    case _ => throw new UnsupportedOperationException(s"can't product over type $typ")
  })

  def apply(v1: Code[_], v2: Code[_]): Code[_] = typ match {
    case _: PInt64 => coerce[Long](v1) * coerce[Long](v2)
    case _: PFloat64 => coerce[Double](v1) * coerce[Double](v2)
    case _ => throw new UnsupportedOperationException(s"can't product over type $typ")
  }
}

class MinAggregator(typ: PType) extends MonoidAggregator(new ComparisonMonoid(typ, "min"))
class MaxAggregator(typ: PType) extends MonoidAggregator(new ComparisonMonoid(typ, "max"))
class SumAggregator(typ: PType) extends MonoidAggregator(new SumMonoid(typ))
class ProductAggregator(typ: PType) extends MonoidAggregator(new ProductMonoid(typ))

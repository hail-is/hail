package is.hail.expr.ir

import is.hail.annotations.aggregators._
import is.hail.asm4s._
import is.hail.expr._
import is.hail.utils._

import scala.reflect.ClassTag

object AggOp {
  private val m: PartialFunction[(AggOp, Type, Type), Code[_] => Code[_]] = {
    case (Fraction(), _: TInt32, _: TInt32) => aggregatorCode[Int](RegionValueIntFractionAggregator)
    case (Fraction(), _: TInt64, _: TInt64) => aggregatorCode[Long](RegionValueLongFractionAggregator)
    case (Fraction(), _: TFloat32, _: TFloat32) => aggregatorCode[Float](RegionValueFloatFractionAggregator)
    case (Fraction(), _: TFloat64, _: TFloat64) => aggregatorCode[Double](RegionValueDoubleFractionAggregator)
      // case (Statistics() =>
      // case (Counter()) =>
      // case (Histogram() =>
      // case (CollectSet() =>
      // case (Collect() =>
      // case (InfoScore() =>
      // case (HardyWeinberg() =>
      // case (Sum() =>
      // case (Product() =>
      // case (Max() =>
      // case (Min() =>
      // case (Take() =>
      // case (TakeBy() =>
  }


  sealed trait AggregatorCodeCurried[T] {
    def apply[Agg >: Null : ClassTag : TypeInfo](t: Type, aggregator: Agg)(implicit tct: ClassTag[T]): AggregatorCode[Agg, T] =
      new AggregatorCode(t, aggregator)
  }

  private object aggregatorCodeCurriedInstance extends AggregatorCodeCurried[Nothing]

  def aggregatorCode[T] = aggregatorCodeCurriedInstance.asInstanceOf[AggregatorCodeCurried[T]]

  class AggregatorCode[Agg >: Null : ClassTag : TypeInfo, T : ClassTag](t: Type, val aggregator: Agg) {
    def seqOp(rva: Code[RegionValueAggregator], v: Code[_], mv: Code[Boolean]): Code[Unit] =
      mv.mux(
        Code.checkcast[Agg](rva).invoke[T, Boolean, Unit]("seqOp", coerce[T](defaultValue(t)), true),
        Code.checkcast[Agg](rva).invoke[T, Boolean, Unit]("seqOp", coerce[T](v), false))
  }
}

sealed trait AggOp { }
final case class Fraction() extends AggOp { } // remove when prefixes work
final case class Statistics() extends AggOp { } // remove when prefixes work
final case class Counter() extends AggOp { }
final case class Histogram() extends AggOp { }
final case class CollectSet() extends AggOp { }
final case class Collect() extends AggOp { }
final case class InfoScore() extends AggOp { }
final case class HardyWeinberg() extends AggOp { } // remove when prefixes work
final case class Sum() extends AggOp { }
final case class Product() extends AggOp { }
final case class Max() extends AggOp { }
final case class Min() extends AggOp { }
final case class Take() extends AggOp { }
final case class TakeBy() extends AggOp { }

// what to do about CallStats
// what to do about InbreedingAggregator

// exists === map(p).sum, needs short-circuiting aggs
// forall === map(p).product, needs short-circuiting aggs
// Count === map(x => 1).sum
// SumArray === Sum, Sum should work on product or union of summables


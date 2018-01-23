package is.hail.expr.ir

import is.hail.annotations.aggregators._
import is.hail.asm4s._
import is.hail.expr.types._
import is.hail.utils._

sealed trait AggOp { }
// nulary
final case class Fraction() extends AggOp { } // remove when prefixes work
final case class Statistics() extends AggOp { } // remove when prefixes work
final case class Collect() extends AggOp { }
// final case class InfoScore() extends AggOp { }
// final case class HardyWeinberg() extends AggOp { } // remove when prefixes work
final case class Sum() extends AggOp { }
// final case class Product() extends AggOp { }
final case class Max() extends AggOp { }
// final case class Min() extends AggOp { }

// unary
final case class Take() extends AggOp { }

// ternary
final case class Histogram() extends AggOp { }


// what to do about CallStats
// what to do about InbreedingAggregator

// exists === map(p).sum, needs short-circuiting aggs
// forall === map(p).product, needs short-circuiting aggs
// Count === map(x => 1).sum
// SumArray === Sum, Sum should work on product or union of summables
// Fraction === map(x => p(x)).fraction
// Statistics === map(x => Cast(x, TDouble())).stats
// Histogram === map(x => Cast(x, TDouble())).hist

// Counter needs Dict
// CollectSet needs Set

// TakeBy needs lambdas

object AggOp {

  def get(op: AggOp, inputType: Type, argumentTypes: Seq[Type]): CodeAggregator[_] =
    m((op, inputType, argumentTypes)).getOrElse(incompatible(op, inputType, argumentTypes))

  def getType(op: AggOp, inputType: Type, argumentTypes: Seq[Type]): Type =
    m((op, inputType, argumentTypes)).getOrElse(incompatible(op, inputType, argumentTypes)).out

  private val m: ((AggOp, Type, Seq[Type])) => Option[CodeAggregator[_]] = lift {
    case (Fraction(), in: TBoolean, Seq()) => CodeAggregator[RegionValueFractionAggregator](in, TFloat64())
    case (Statistics(), in: TFloat64, Seq()) => CodeAggregator[RegionValueStatisticsAggregator](in, RegionValueStatisticsAggregator.typ)
    case (Collect(), in: TBoolean, Seq()) => CodeAggregator[RegionValueCollectBooleanAggregator](in, TArray(TBoolean()))
    case (Collect(), in: TInt32, Seq()) => CodeAggregator[RegionValueCollectIntAggregator](in, TArray(TInt32()))
    // FIXME: implement these
    // case (Collect(), _: TInt64) =>
    // case (Collect(), _: TFloat32) =>
    // case (Collect(), _: TFloat64) =>
    // case (Collect(), _: TArray) =>
    // case (Collect(), _: TStruct) =>
    // case (InfoScore() =>
    case (Sum(), in: TInt32, Seq()) => CodeAggregator[RegionValueSumIntAggregator](in, TInt32())
    case (Sum(), in: TInt64, Seq()) => CodeAggregator[RegionValueSumLongAggregator](in, TInt64())
    case (Sum(), in: TFloat32, Seq()) => CodeAggregator[RegionValueSumFloatAggregator](in, TFloat32())
    case (Sum(), in: TFloat64, Seq()) => CodeAggregator[RegionValueSumDoubleAggregator](in, TFloat64())
    // case (HardyWeinberg(), _: T) =>
    // case (Product(), _: T) =>
    case (Max(), in: TBoolean, Seq()) => CodeAggregator[RegionValueMaxBooleanAggregator](in, TBoolean())
    case (Max(), in: TInt32, Seq()) => CodeAggregator[RegionValueMaxIntAggregator](in, TInt32())
    case (Max(), in: TInt64, Seq()) => CodeAggregator[RegionValueMaxLongAggregator](in, TInt64())
    case (Max(), in: TFloat32, Seq()) => CodeAggregator[RegionValueMaxFloatAggregator](in, TFloat32())
    case (Max(), in: TFloat64, Seq()) => CodeAggregator[RegionValueMaxDoubleAggregator](in, TFloat64())
    // case (Min(), _: T) =>
    case (Take(), in: TBoolean, args@Seq(_: TInt32)) => CodeAggregator[RegionValueTakeBooleanAggregator](in, TArray(in), classOf[Int])
    case (Take(), in: TInt32, args@Seq(_: TInt32)) => CodeAggregator[RegionValueTakeIntAggregator](in, TArray(in), classOf[Int])
    case (Take(), in: TInt64, args@Seq(_: TInt32)) => CodeAggregator[RegionValueTakeLongAggregator](in, TArray(in), classOf[Int])
    case (Take(), in: TFloat32, args@Seq(_: TInt32)) => CodeAggregator[RegionValueTakeFloatAggregator](in, TArray(in), classOf[Int])
    case (Take(), in: TFloat64, args@Seq(_: TInt32)) => CodeAggregator[RegionValueTakeDoubleAggregator](in, TArray(in), classOf[Int])
    case (Histogram(), in: TFloat64, args@Seq(_: TFloat64, _: TFloat64, _: TInt32)) =>
      CodeAggregator[RegionValueHistogramAggregator](in, RegionValueHistogramAggregator.typ, classOf[Double], classOf[Double], classOf[Int])
  }

  private def incompatible(op: AggOp, inputType: Type, argumentTypes: Seq[Type]): Nothing = {
    throw new RuntimeException(s"no aggregator named $op taking arguments (${argumentTypes.mkString(", ")}) operating on aggregables of type $inputType")
  }
}

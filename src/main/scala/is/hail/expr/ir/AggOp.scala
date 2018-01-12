package is.hail.expr.ir

import is.hail.annotations.aggregators._
import is.hail.asm4s._
import is.hail.expr.types._
import is.hail.utils._

sealed trait AggOp { }
final case class Fraction() extends AggOp { } // remove when prefixes work
final case class Statistics() extends AggOp { } // remove when prefixes work
final case class Collect() extends AggOp { }
// final case class InfoScore() extends AggOp { }
// final case class HardyWeinberg() extends AggOp { } // remove when prefixes work
final case class Sum() extends AggOp { }
// final case class Product() extends AggOp { }
final case class Max() extends AggOp { }
// final case class Min() extends AggOp { }

sealed trait AggOp { }
final case class Take() extends AggOp { }

sealed trait AggOp { }
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

  def get(op: AggOp, r: Type): CodeAggregator =
    m((op, r)).getOrElse(incompatible(op, r))

  def getType(op: AggOp, r: Type): Type =
    m((op, r)).getOrElse(incompatible(op, r)).out

  private val m: ((AggOp, Array[Type])) => Option[CodeAggregator] = lift {
    case (Fraction(), Array(_: TBoolean)) => CodeAggregator.nullary[Boolean](new RegionValueFractionAggregator(), TFloat64())
    case (Statistics(), Array(_: TFloat64)) => CodeAggregator.nullary[Double](new RegionValueStatisticsAggregator(), RegionValueStatisticsAggregator.typ)
    case (Collect(), Array(_: TBoolean)) => CodeAggregator.nullary[Boolean](new RegionValueCollectBooleanAggregator(), TArray(TBoolean()))
    case (Collect(), Array(_: TInt32)) => CodeAggregator.nullary[Int](new RegionValueCollectIntAggregator(), TArray(TInt32()))
    // FIXME: implement these
    // case (Collect(), _: TInt64) => CodeAggregator.nullary[Long](new RegionValueCollectLongAggregator())
    // case (Collect(), _: TFloat32) => CodeAggregator.nullary[Float](new RegionValueCollectFloatAggregator())
    // case (Collect(), _: TFloat64) => CodeAggregator.nullary[Double](new RegionValueCollectDoubleAggregator())
    // case (Collect(), _: TArray) => CodeAggregator.nullary[Long](new RegionValueCollectArrayAggregator())
    // case (Collect(), _: TStruct) => CodeAggregator.nullary[Long](new RegionValueCollectStructAggregator())
    // case (InfoScore() =>
    case (Sum(), Array(_: TInt32)) => CodeAggregator.nullary[Int](new RegionValueSumIntAggregator(), TInt32())
    case (Sum(), Array(_: TInt64)) => CodeAggregator.nullary[Long](new RegionValueSumLongAggregator(), TInt64())
    case (Sum(), Array(_: TFloat32)) => CodeAggregator.nullary[Float](new RegionValueSumFloatAggregator(), TFloat32())
    case (Sum(), Array(_: TFloat64)) => CodeAggregator.nullary[Double](new RegionValueSumDoubleAggregator(), TFloat64())
    // case (HardyWeinberg(), _: T) =>
    // case (Product(), _: T) =>
    case (Max(), Array(_: TBoolean)) => CodeAggregator.nullary[Boolean](new RegionValueMaxBooleanAggregator(), TBoolean())
    case (Max(), Array(_: TInt32)) => CodeAggregator.nullary[Int](new RegionValueMaxIntAggregator(), TInt32())
    case (Max(), Array(_: TInt64)) => CodeAggregator.nullary[Long](new RegionValueMaxLongAggregator(), TInt64())
    case (Max(), Array(_: TFloat32)) => CodeAggregator.nullary[Float](new RegionValueMaxFloatAggregator(), TFloat32())
    case (Max(), Array(_: TFloat64)) => CodeAggregator.nullary[Double](new RegionValueMaxDoubleAggregator(), TFloat64())
    // case (Min(), _: T) =>
    case (Take(), Array(_: TInt32, _: TBoolean)) => CodeAggregator.unary[Int, Boolean](RegionValueTakeBooleanAggregator.stagedNew, TArray(TBoolean()))
    case (Take(), Array(_: TInt32, _: TInt32)) => CodeAggregator.unary[Int, Int](RegionValueTakeIntAggregator.stagedNew, TArray(TInt32()))
    case (Take(), Array(_: TInt32, _: TInt64)) => CodeAggregator.unary[Int, Long](RegionValueTakeLongAggregator.stagedNew, TArray(TInt64()))
    case (Take(), Array(_: TInt32, _: TFloat32)) => CodeAggregator.unary[Int, Float](RegionValueTakeFloatAggregator.stagedNew, TArray(TFloat32()))
    case (Take(), Array(_: TInt32, _: TFloat64)) => CodeAggregator.unary[Int, Double](RegionValueTakeDoubleAggregator.stagedNew, TArray(TFloat64()))
    case (Histogram(), Array(_: TFloat64, _: TFloat64, _: TInt32, _: TFloat64)) =>
      CodeAggregator.ternary[Double, Double, Int, Double](RegionValueHistogramAggregator.stagedNew, RegionValueHistogramAggregator.typ)
  }

  private def incompatible(op: AggOp, r: Type): Nothing =
    throw new RuntimeException(s"no aggregator named $op operating on aggregables of type $r")

  private def incompatible(op: AggOp, t: Type, r: Type): Nothing =
    throw new RuntimeException(s"no aggregator named $op with arguments of type $t, operating on aggregables of type $r")

  private def incompatible(op: AggOp, t: Type, u: Type, v: Type, r: Type): Nothing =
    throw new RuntimeException(s"no aggregator named $op with arguments of type $t, $u, and $v operating on aggregables of type $r")
}

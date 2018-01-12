package is.hail.expr.ir

import is.hail.annotations.aggregators._
import is.hail.asm4s._
import is.hail.expr.types._
import is.hail.utils._

sealed trait NullaryAggOp { }
final case class Fraction() extends NullaryAggOp { } // remove when prefixes work
final case class Statistics() extends NullaryAggOp { } // remove when prefixes work
final case class Collect() extends NullaryAggOp { }
// final case class InfoScore() extends NullaryAggOp { }
// final case class HardyWeinberg() extends NullaryAggOp { } // remove when prefixes work
final case class Sum() extends NullaryAggOp { }
// final case class Product() extends NullaryAggOp { }
final case class Max() extends NullaryAggOp { }
// final case class Min() extends NullaryAggOp { }

sealed trait UnaryAggOp { }
final case class Take() extends UnaryAggOp { }

sealed trait TernaryAggOp { }
final case class Histogram() extends TernaryAggOp { }


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

  def getNullary(op: NullaryAggOp, r: Type): CodeAggregator.Nullary =
    nullary((op, r)).getOrElse(incompatible(op, r))

  def getNullaryType(op: NullaryAggOp, r: Type): Type =
    nullary((op, r)).getOrElse(incompatible(op, r)).out

  def getUnary(op: UnaryAggOp, t: Type, r: Type): CodeAggregator.Unary =
    unary((op, t, r)).getOrElse(incompatible(op, t, r))

  def getUnaryType(op: UnaryAggOp, t: Type, r: Type): Type =
    unary((op, t, r)).getOrElse(incompatible(op, t, r)).out

  def getTernary(op: TernaryAggOp, t: Type, u: Type, v: Type, r: Type): CodeAggregator.Ternary =
    ternary((op, t, u, v, r)).getOrElse(incompatible(op, t, u, v, r))

  def getTernaryType(op: TernaryAggOp, t: Type, u: Type, v: Type, r: Type): Type =
    ternary((op, t, u, v, r)).getOrElse(incompatible(op, t, u, v, r)).out

  private val nullary: ((NullaryAggOp, Type)) => Option[CodeAggregator.Nullary] = lift {
    case (Fraction(), _: TBoolean) => CodeAggregator.nullary[Boolean](new RegionValueFractionAggregator(), TFloat64())
    case (Statistics(), _: TFloat64) => CodeAggregator.nullary[Double](new RegionValueStatisticsAggregator(), RegionValueStatisticsAggregator.typ)
    case (Collect(), _: TBoolean) => CodeAggregator.nullary[Boolean](new RegionValueCollectBooleanAggregator(), TArray(TBoolean()))
    case (Collect(), _: TInt32) => CodeAggregator.nullary[Int](new RegionValueCollectIntAggregator(), TArray(TInt32()))
    // FIXME: implement these
    // case (Collect(), _: TInt64) => CodeAggregator.nullary[Long](new RegionValueCollectLongAggregator())
    // case (Collect(), _: TFloat32) => CodeAggregator.nullary[Float](new RegionValueCollectFloatAggregator())
    // case (Collect(), _: TFloat64) => CodeAggregator.nullary[Double](new RegionValueCollectDoubleAggregator())
    // case (Collect(), _: TArray) => CodeAggregator.nullary[Long](new RegionValueCollectArrayAggregator())
    // case (Collect(), _: TStruct) => CodeAggregator.nullary[Long](new RegionValueCollectStructAggregator())
    // case (InfoScore() =>
    case (Sum(), _: TInt32) => CodeAggregator.nullary[Int](new RegionValueSumIntAggregator(), TInt32())
    case (Sum(), _: TInt64) => CodeAggregator.nullary[Long](new RegionValueSumLongAggregator(), TInt64())
    case (Sum(), _: TFloat32) => CodeAggregator.nullary[Float](new RegionValueSumFloatAggregator(), TFloat32())
    case (Sum(), _: TFloat64) => CodeAggregator.nullary[Double](new RegionValueSumDoubleAggregator(), TFloat64())
    // case (HardyWeinberg(), _: T) =>
    // case (Product(), _: T) =>
    case (Max(), _: TBoolean) => CodeAggregator.nullary[Boolean](new RegionValueMaxBooleanAggregator(), TBoolean())
    case (Max(), _: TInt32) => CodeAggregator.nullary[Int](new RegionValueMaxIntAggregator(), TInt32())
    case (Max(), _: TInt64) => CodeAggregator.nullary[Long](new RegionValueMaxLongAggregator(), TInt64())
    case (Max(), _: TFloat32) => CodeAggregator.nullary[Float](new RegionValueMaxFloatAggregator(), TFloat32())
    case (Max(), _: TFloat64) => CodeAggregator.nullary[Double](new RegionValueMaxDoubleAggregator(), TFloat64())
    // case (Min(), _: T) =>
  }

  private val unary: ((UnaryAggOp, Type, Type)) => Option[CodeAggregator.Unary] = lift {
    case (Take(), _: TInt32, _: TBoolean) => CodeAggregator.unary[Int, Boolean](RegionValueTakeBooleanAggregator.stagedNew, TArray(TBoolean()))
    case (Take(), _: TInt32, _: TInt32) => CodeAggregator.unary[Int, Int](RegionValueTakeIntAggregator.stagedNew, TArray(TInt32()))
    case (Take(), _: TInt32, _: TInt64) => CodeAggregator.unary[Int, Long](RegionValueTakeLongAggregator.stagedNew, TArray(TInt64()))
    case (Take(), _: TInt32, _: TFloat32) => CodeAggregator.unary[Int, Float](RegionValueTakeFloatAggregator.stagedNew, TArray(TFloat32()))
    case (Take(), _: TInt32, _: TFloat64) => CodeAggregator.unary[Int, Double](RegionValueTakeDoubleAggregator.stagedNew, TArray(TFloat64()))
  }

  private val ternary: ((TernaryAggOp, Type, Type, Type, Type)) => Option[CodeAggregator.Ternary] = lift {
    case (Histogram(), _: TFloat64, _: TFloat64, _: TInt32, _: TFloat64) =>
      CodeAggregator.ternary[Double, Double, Int, Double](RegionValueHistogramAggregator.stagedNew, RegionValueHistogramAggregator.typ)
  }

  private def incompatible(op: NullaryAggOp, r: Type): Nothing =
    throw new RuntimeException(s"no aggregator named $op operating on aggregables of type $r")

  private def incompatible(op: UnaryAggOp, t: Type, r: Type): Nothing =
    throw new RuntimeException(s"no aggregator named $op with arguments of type $t, operating on aggregables of type $r")

  private def incompatible(op: TernaryAggOp, t: Type, u: Type, v: Type, r: Type): Nothing =
    throw new RuntimeException(s"no aggregator named $op with arguments of type $t, $u, and $v operating on aggregables of type $r")
}

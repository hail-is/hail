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

  def get(op: AggOp, typs: Array[Type]): CodeAggregator[_, _] =
    m((op, typs)).getOrElse(incompatible(op, typs))

  def getType(op: AggOp, typs: Array[Type]): Type =
    m((op, typs)).getOrElse(incompatible(op, typs)).out

  private val m: ((AggOp, Array[Type])) => Option[CodeAggregator[_, _]] = lift {
    case (Fraction(), Array(_: TBoolean)) => CodeAggregator[Boolean]((_,_) => Code.newInstance[RegionValueFractionAggregator](), TFloat64())
    case (Statistics(), Array(_: TFloat64)) => CodeAggregator[Double]((_,_) => Code.newInstance[RegionValueStatisticsAggregator](), RegionValueStatisticsAggregator.typ)
    case (Collect(), Array(_: TBoolean)) => CodeAggregator[Boolean]((_,_) => Code.newInstance[RegionValueCollectBooleanAggregator](), TArray(TBoolean()))
    case (Collect(), Array(_: TInt32)) => CodeAggregator[Int]((_,_) => Code.newInstance[RegionValueCollectIntAggregator](), TArray(TInt32()))
    // FIXME: implement these
    // case (Collect(), _: TInt64) => CodeAggregator[Long]((_,_) => Code.newInstance[RegionValueCollectLongAggregator]())
    // case (Collect(), _: TFloat32) => CodeAggregator[Float]((_,_) => Code.newInstance[RegionValueCollectFloatAggregator]())
    // case (Collect(), _: TFloat64) => CodeAggregator[Double]((_,_) => Code.newInstance[RegionValueCollectDoubleAggregator]())
    // case (Collect(), _: TArray) => CodeAggregator[Long]((_,_) => Code.newInstance[RegionValueCollectArrayAggregator]())
    // case (Collect(), _: TStruct) => CodeAggregator[Long]((_,_) => Code.newInstance[RegionValueCollectStructAggregator]())
    // case (InfoScore() =>
    case (Sum(), Array(_: TInt32)) => CodeAggregator[Int]((_,_) => Code.newInstance[RegionValueSumIntAggregator](), TInt32())
    case (Sum(), Array(_: TInt64)) => CodeAggregator[Long]((_,_) => Code.newInstance[RegionValueSumLongAggregator](), TInt64())
    case (Sum(), Array(_: TFloat32)) => CodeAggregator[Float]((_,_) => Code.newInstance[RegionValueSumFloatAggregator](), TFloat32())
    case (Sum(), Array(_: TFloat64)) => CodeAggregator[Double]((_,_) => Code.newInstance[RegionValueSumDoubleAggregator](), TFloat64())
    // case (HardyWeinberg(), _: T) =>
    // case (Product(), _: T) =>
    case (Max(), Array(_: TBoolean)) => CodeAggregator[Boolean]((_,_) => Code.newInstance[RegionValueMaxBooleanAggregator](), TBoolean())
    case (Max(), Array(_: TInt32)) => CodeAggregator[Int]((_,_) => Code.newInstance[RegionValueMaxIntAggregator](), TInt32())
    case (Max(), Array(_: TInt64)) => CodeAggregator[Long]((_,_) => Code.newInstance[RegionValueMaxLongAggregator](), TInt64())
    case (Max(), Array(_: TFloat32)) => CodeAggregator[Float]((_,_) => Code.newInstance[RegionValueMaxFloatAggregator](), TFloat32())
    case (Max(), Array(_: TFloat64)) => CodeAggregator[Double]((_,_) => Code.newInstance[RegionValueMaxDoubleAggregator](), TFloat64())
    // case (Min(), _: T) =>
    case (Take(), Array(_: TInt32, _: TBoolean)) => CodeAggregator[Int](RegionValueTakeBooleanAggregator.stagedNew, TArray(TBoolean()))
    case (Take(), Array(_: TInt32, _: TInt32)) => CodeAggregator[Int](RegionValueTakeIntAggregator.stagedNew, TArray(TInt32()))
    case (Take(), Array(_: TInt32, _: TInt64)) => CodeAggregator[Int](RegionValueTakeLongAggregator.stagedNew, TArray(TInt64()))
    case (Take(), Array(_: TInt32, _: TFloat32)) => CodeAggregator[Int](RegionValueTakeFloatAggregator.stagedNew, TArray(TFloat32()))
    case (Take(), Array(_: TInt32, _: TFloat64)) => CodeAggregator[Int](RegionValueTakeDoubleAggregator.stagedNew, TArray(TFloat64()))
    case (Histogram(), Array(_: TFloat64, _: TFloat64, _: TInt32, _: TFloat64)) =>
      CodeAggregator[Double](RegionValueHistogramAggregator.stagedNew, RegionValueHistogramAggregator.typ)
  }

  private def incompatible(op: AggOp, typs: Array[Type]): Nothing = {
    val typs2 = typs.reverse
    val elementType = typs2.head
    val argumentTypes = typs2.tail.reverse
    throw new RuntimeException(s"no aggregator named $op taking arguments (${argumentTypes.mkString(",")}) operating on aggregables of type $elementType")
  }
}

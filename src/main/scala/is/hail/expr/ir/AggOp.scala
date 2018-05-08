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
final case class Product() extends AggOp { }
final case class Max() extends AggOp { }
final case class Min() extends AggOp { }

final case class Count() extends AggOp { }

// unary
final case class Take() extends AggOp { }

// ternary
final case class Histogram() extends AggOp { }


// what to do about CallStats
final case class CallStats() extends AggOp { }
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

  def get(op: AggOp, inputType: Type, constructorTypes: Seq[Type], initOpTypes: Option[Seq[Type]]): CodeAggregator[T] forSome { type T <: RegionValueAggregator } =
    m((op, inputType, constructorTypes, initOpTypes)).getOrElse(incompatible(op, inputType, constructorTypes, initOpTypes))

  def getType(op: AggOp, inputType: Type, constructorTypes: Seq[Type], initOpTypes: Option[Seq[Type]] = None): Type =
    m((op, inputType, constructorTypes, initOpTypes)).getOrElse(incompatible(op, inputType, constructorTypes, initOpTypes)).out

  private val m: ((AggOp, Type, Seq[Type], Option[Seq[Type]])) => Option[CodeAggregator[T] forSome { type T <: RegionValueAggregator }] = lift {
    case (Fraction(), in: TBoolean, Seq(), None) => CodeAggregator[RegionValueFractionAggregator](in, TArray(TFloat64()))
    case (Statistics(), in: TFloat64, Seq(), None) => CodeAggregator[RegionValueStatisticsAggregator](in, RegionValueStatisticsAggregator.typ)
    case (Collect(), in: TBoolean, Seq(), None) => CodeAggregator[RegionValueCollectBooleanAggregator](in, TArray(TBoolean()))
    case (Collect(), in: TInt32, Seq(), None) => CodeAggregator[RegionValueCollectIntAggregator](in, TArray(TInt32()))
    // FIXME: implement these
    // case (Collect(), _: TInt64) =>
    // case (Collect(), _: TFloat32) =>
    // case (Collect(), _: TFloat64) =>
    // case (Collect(), _: TArray) =>
    // case (Collect(), _: TStruct) =>
    // case (InfoScore() =>

    case (Sum(), in: TInt64, Seq(), None) => CodeAggregator[RegionValueSumLongAggregator](in, TInt64())
    case (Sum(), in: TFloat64, Seq(), None) => CodeAggregator[RegionValueSumDoubleAggregator](in, TFloat64())

    case (Product(), in: TInt64, Seq(), None) => CodeAggregator[RegionValueProductLongAggregator](in, TInt64())
    case (Product(), in: TFloat64, Seq(), None) => CodeAggregator[RegionValueProductDoubleAggregator](in, TFloat64())

    // case (HardyWeinberg(), _: T) =>

    case (Max(), in: TBoolean, Seq(), None) => CodeAggregator[RegionValueMaxBooleanAggregator](in, TBoolean())
    case (Max(), in: TInt32, Seq(), None) => CodeAggregator[RegionValueMaxIntAggregator](in, TInt32())
    case (Max(), in: TInt64, Seq(), None) => CodeAggregator[RegionValueMaxLongAggregator](in, TInt64())
    case (Max(), in: TFloat32, Seq(), None) => CodeAggregator[RegionValueMaxFloatAggregator](in, TFloat32())
    case (Max(), in: TFloat64, Seq(), None) => CodeAggregator[RegionValueMaxDoubleAggregator](in, TFloat64())

    case (Min(), in: TBoolean, Seq(), None) => CodeAggregator[RegionValueMinBooleanAggregator](in, TBoolean())
    case (Min(), in: TInt32, Seq(), None) => CodeAggregator[RegionValueMinIntAggregator](in, TInt32())
    case (Min(), in: TInt64, Seq(), None) => CodeAggregator[RegionValueMinLongAggregator](in, TInt64())
    case (Min(), in: TFloat32, Seq(), None) => CodeAggregator[RegionValueMinFloatAggregator](in, TFloat32())
    case (Min(), in: TFloat64, Seq(), None) => CodeAggregator[RegionValueMinDoubleAggregator](in, TFloat64())

    case (Count(), in, Seq(), None) => CodeAggregator[RegionValueCountAggregator](in, TInt64())

    case (Take(), in: TBoolean, constArgs@Seq(_: TInt32), None) => CodeAggregator[RegionValueTakeBooleanAggregator](in, TArray(in), constrArgTypes = Array(classOf[Int]))
    case (Take(), in: TInt32, constArgs@Seq(_: TInt32), None) => CodeAggregator[RegionValueTakeIntAggregator](in, TArray(in), constrArgTypes = Array(classOf[Int]))
    case (Take(), in: TInt64, constArgs@Seq(_: TInt32), None) => CodeAggregator[RegionValueTakeLongAggregator](in, TArray(in), constrArgTypes = Array(classOf[Int]))
    case (Take(), in: TFloat32, constArgs@Seq(_: TInt32), None) => CodeAggregator[RegionValueTakeFloatAggregator](in, TArray(in), constrArgTypes = Array(classOf[Int]))
    case (Take(), in: TFloat64, constArgs@Seq(_: TInt32), None) => CodeAggregator[RegionValueTakeDoubleAggregator](in, TArray(in), constrArgTypes = Array(classOf[Int]))

    case (Histogram(), in: TFloat64, constArgs@Seq(_: TFloat64, _: TFloat64, _: TInt32), None) =>
      CodeAggregator[RegionValueHistogramAggregator](in, RegionValueHistogramAggregator.typ, constrArgTypes = Array(classOf[Double], classOf[Double], classOf[Int]))

    case (CallStats(), in: TCall, Seq(), initOpArgs@Some(Seq(TArray(_: TString, _)))) =>
      CodeAggregator[RegionValueCallStatsAggregator](in, RegionValueCallStatsAggregator.typ, initOpArgTypes = Some(Array(classOf[Long])))
  }

  private def incompatible(op: AggOp, inputType: Type, constTypes: Seq[Type], initOpTypes: Option[Seq[Type]]): Nothing = {
    throw new RuntimeException(s"no aggregator named $op taking arguments ([${ constTypes.mkString(", ") }]" +
      s", ${ if (initOpTypes.isEmpty) "None" else "[" + initOpTypes.get.mkString(", ") + "]" }) " +
      s"operating on aggregables of type $inputType")
  }
}

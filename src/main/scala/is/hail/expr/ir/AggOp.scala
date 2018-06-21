package is.hail.expr.ir

import is.hail.annotations.aggregators._
import is.hail.asm4s._
import is.hail.expr.types._
import is.hail.utils._

case class AggSignature(
  op: AggOp,
  inputType: Type,
  constructorArgs: Seq[Type],
  initOpArgs: Option[Seq[Type]],
  seqOpArgs: Seq[Type]
)

sealed trait AggOp { }
// nulary
final case class Fraction() extends AggOp { } // remove when prefixes work
final case class Statistics() extends AggOp { } // remove when prefixes work
final case class Collect() extends AggOp { }
final case class CollectAsSet() extends AggOp { }
final case class InfoScore() extends AggOp { }
final case class HardyWeinberg() extends AggOp { } // remove when prefixes work

final case class Sum() extends AggOp { }
final case class Product() extends AggOp { }
final case class Max() extends AggOp { }
final case class Min() extends AggOp { }

final case class Count() extends AggOp { }

// unary
final case class Take() extends AggOp { }

// ternary
final case class Histogram() extends AggOp { }

final case class CallStats() extends AggOp { }

// what to do about InbreedingAggregator
final case class Inbreeding() extends AggOp { }

// exists === map(p).sum, needs short-circuiting aggs
// forall === map(p).product, needs short-circuiting aggs
// Count === map(x => 1).sum
// SumArray === Sum, Sum should work on product or union of summables
// Fraction === map(x => p(x)).fraction
// Statistics === map(x => Cast(x, TDouble())).stats
// Histogram === map(x => Cast(x, TDouble())).hist

// Counter needs Dict
final case class Counter() extends AggOp { }
// CollectSet needs Set

// TakeBy needs lambdas

object AggOp {

  def get(aggSig: AggSignature): CodeAggregator[T] forSome { type T <: RegionValueAggregator } =
    getOption(aggSig).getOrElse(incompatible(aggSig))

  def getType(aggSig: AggSignature): Type = getOption(aggSig).getOrElse(incompatible(aggSig)).out

  def getOption(aggSig: AggSignature): Option[CodeAggregator[T] forSome { type T <: RegionValueAggregator }] =
    getOption(aggSig.op, aggSig.inputType, aggSig.constructorArgs, aggSig.initOpArgs, aggSig.seqOpArgs)

  val getOption: ((AggOp, Type, Seq[Type], Option[Seq[Type]], Seq[Type])) => Option[CodeAggregator[T] forSome { type T <: RegionValueAggregator }] = lift {
    case (Fraction(), in: TBoolean, Seq(), None, Seq()) => CodeAggregator[RegionValueFractionAggregator](in, TFloat64())

    case (Statistics(), in: TFloat64, Seq(), None, Seq()) => CodeAggregator[RegionValueStatisticsAggregator](in, RegionValueStatisticsAggregator.typ)

    case (Collect(), in, Seq(), None, Seq()) => in match {
      case _: TBoolean => CodeAggregator[RegionValueCollectBooleanAggregator](in, TArray(in))
      case _: TInt32 | _: TCall => CodeAggregator[RegionValueCollectIntAggregator](in, TArray(in))
      case _: TInt64 => CodeAggregator[RegionValueCollectLongAggregator](in, TArray(in))
      case _: TFloat32 => CodeAggregator[RegionValueCollectFloatAggregator](in, TArray(in))
      case _: TFloat64 => CodeAggregator[RegionValueCollectDoubleAggregator](in, TArray(in))
      case _ => CodeAggregator[RegionValueCollectAnnotationAggregator](in, TArray(in), constrArgTypes = Array(classOf[Type]))
    }

    case (InfoScore(), in@TArray(TFloat64(_), _), Seq(), None, Seq()) => CodeAggregator[RegionValueInfoScoreAggregator](in, RegionValueInfoScoreAggregator.typ)
      
    case (Sum(), in: TInt64, Seq(), None, Seq()) => CodeAggregator[RegionValueSumLongAggregator](in, TInt64())
    case (Sum(), in: TFloat64, Seq(), None, Seq()) => CodeAggregator[RegionValueSumDoubleAggregator](in, TFloat64())
    case (Sum(), in@TArray(TInt64(_), _), Seq(), None, Seq()) =>
      CodeAggregator[RegionValueArraySumLongAggregator](in, TArray(TInt64()))

    case (CollectAsSet(), in, Seq(), None, Seq()) =>
      in match {
        case _: TBoolean => CodeAggregator[RegionValueCollectAsSetBooleanAggregator](in, TSet(TBoolean()))
        case _: TInt32 => CodeAggregator[RegionValueCollectAsSetIntAggregator](in, TSet(TInt32()))
        case _: TInt64 => CodeAggregator[RegionValueCollectAsSetLongAggregator](in, TSet(TInt64()))
        case _: TFloat32 => CodeAggregator[RegionValueCollectAsSetFloatAggregator](in, TSet(TFloat32()))
        case _: TFloat64 => CodeAggregator[RegionValueCollectAsSetDoubleAggregator](in, TSet(TFloat64()))
        case _ => CodeAggregator[RegionValueCollectAsSetAnnotationAggregator](in, TSet(in), constrArgTypes = Array(classOf[Type]))
      }
    case (Sum(), in@TArray(TFloat64(_), _), Seq(), None, Seq()) =>
      CodeAggregator[RegionValueArraySumDoubleAggregator](in, TArray(TFloat64()))

    case (Product(), in: TInt64, Seq(), None, Seq()) => CodeAggregator[RegionValueProductLongAggregator](in, TInt64())
    case (Product(), in: TFloat64, Seq(), None, Seq()) => CodeAggregator[RegionValueProductDoubleAggregator](in, TFloat64())

    case (HardyWeinberg(), in: TCall, Seq(), None, Seq()) => CodeAggregator[RegionValueHardyWeinbergAggregator](in, RegionValueHardyWeinbergAggregator.typ)

    case (Max(), in: TBoolean, Seq(), None, Seq()) => CodeAggregator[RegionValueMaxBooleanAggregator](in, TBoolean())
    case (Max(), in: TInt32, Seq(), None, Seq()) => CodeAggregator[RegionValueMaxIntAggregator](in, TInt32())
    case (Max(), in: TInt64, Seq(), None, Seq()) => CodeAggregator[RegionValueMaxLongAggregator](in, TInt64())
    case (Max(), in: TFloat32, Seq(), None, Seq()) => CodeAggregator[RegionValueMaxFloatAggregator](in, TFloat32())
    case (Max(), in: TFloat64, Seq(), None, Seq()) => CodeAggregator[RegionValueMaxDoubleAggregator](in, TFloat64())

    case (Min(), in: TBoolean, Seq(), None, Seq()) => CodeAggregator[RegionValueMinBooleanAggregator](in, TBoolean())
    case (Min(), in: TInt32, Seq(), None, Seq()) => CodeAggregator[RegionValueMinIntAggregator](in, TInt32())
    case (Min(), in: TInt64, Seq(), None, Seq()) => CodeAggregator[RegionValueMinLongAggregator](in, TInt64())
    case (Min(), in: TFloat32, Seq(), None, Seq()) => CodeAggregator[RegionValueMinFloatAggregator](in, TFloat32())
    case (Min(), in: TFloat64, Seq(), None, Seq()) => CodeAggregator[RegionValueMinDoubleAggregator](in, TFloat64())

    case (Count(), in, Seq(), None, Seq()) => CodeAggregator[RegionValueCountAggregator](in, TInt64())

    case (Counter(), in, Seq(), None, Seq()) => in match {
      case _: TBoolean => CodeAggregator[RegionValueCounterBooleanAggregator](in, TDict(in, TInt64()))
      case _: TInt32 | _: TCall => CodeAggregator[RegionValueCounterIntAggregator](in, TDict(in, TInt64()), constrArgTypes = Array(classOf[Type]))
      case _: TInt64 => CodeAggregator[RegionValueCounterLongAggregator](in, TDict(in, TInt64()), constrArgTypes = Array(classOf[Type]))
      case _: TFloat32 => CodeAggregator[RegionValueCounterFloatAggregator](in, TDict(in, TInt64()), constrArgTypes = Array(classOf[Type]))
      case _: TFloat64 => CodeAggregator[RegionValueCounterDoubleAggregator](in, TDict(in, TInt64()), constrArgTypes = Array(classOf[Type]))
      case _ => CodeAggregator[RegionValueCounterAnnotationAggregator](in, TDict(in, TInt64()), constrArgTypes = Array(classOf[Type]))
    }

    case (Take(), in, constArgs@Seq(_: TInt32), None, Seq()) => in match {
      case _: TBoolean => CodeAggregator[RegionValueTakeBooleanAggregator](in, TArray(in), constrArgTypes = Array(classOf[Int]))
      case _: TInt32 | _: TCall => CodeAggregator[RegionValueTakeIntAggregator](in, TArray(in), constrArgTypes = Array(classOf[Int]))
      case _: TInt64 => CodeAggregator[RegionValueTakeLongAggregator](in, TArray(in), constrArgTypes = Array(classOf[Int]))
      case _: TFloat32 => CodeAggregator[RegionValueTakeFloatAggregator](in, TArray(in), constrArgTypes = Array(classOf[Int]))
      case _: TFloat64 => CodeAggregator[RegionValueTakeDoubleAggregator](in, TArray(in), constrArgTypes = Array(classOf[Int]))
      case _ => CodeAggregator[RegionValueTakeAnnotationAggregator](in, TArray(in), constrArgTypes = Array(classOf[Int], classOf[Type]))
    }

    case (Histogram(), in: TFloat64, constArgs@Seq(_: TFloat64, _: TFloat64, _: TInt32), None, Seq()) =>
      CodeAggregator[RegionValueHistogramAggregator](in, RegionValueHistogramAggregator.typ, constrArgTypes = Array(classOf[Double], classOf[Double], classOf[Int]))

    case (CallStats(), in: TCall, Seq(), initOpArgs@Some(Seq(_: TInt32)), Seq()) =>
      CodeAggregator[RegionValueCallStatsAggregator](in, RegionValueCallStatsAggregator.typ, initOpArgTypes = Some(Array(classOf[Int])))

    case (Inbreeding(), in: TCall, Seq(), None, seqOpArgs@Seq(_: TFloat64)) =>
      CodeAggregator[RegionValueInbreedingAggregator](in, RegionValueInbreedingAggregator.typ, seqOpArgTypes = Array(classOf[Double]))
  }

  private def incompatible(aggSig: AggSignature): Nothing = {
    throw new RuntimeException(s"no aggregator named ${ aggSig.op } taking arguments ([${ aggSig.constructorArgs.mkString(", ") }]" +
      s", ${ if (aggSig.initOpArgs.isEmpty) "None" else "[" + aggSig.initOpArgs.get.mkString(", ") + "]" }) " +
      s"operating on aggregables of type ${ aggSig.inputType }")
  }

  val fromString: PartialFunction[String, AggOp] = {
    case "fraction" => Fraction()
    case "stats" => Statistics()
    case "collect" => Collect()
    case "sum" => Sum()
    case "product" => Product()
    case "max" => Max()
    case "min" => Min()
    case "count" => Count()
    case "counter" => Counter()
    case "take" => Take()
    case "hist" => Histogram()
    case "infoScore" => InfoScore()
    case "callStats" => CallStats()
    case "inbreeding" => Inbreeding()
    case "hardyWeinberg" => HardyWeinberg()
  }
}

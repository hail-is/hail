package is.hail.expr.ir

import is.hail.annotations.aggregators._
import is.hail.asm4s._
import is.hail.expr.types._
import is.hail.utils._

import scala.reflect.ClassTag

case class AggSignature(
  op: AggOp,
  constructorArgs: Seq[Type],
  initOpArgs: Option[Seq[Type]],
  seqOpArgs: Seq[Type])

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
final case class TakeBy() extends AggOp { }

object AggOp {

  def get(aggSig: AggSignature): CodeAggregator[T] forSome { type T <: RegionValueAggregator } =
    getOption(aggSig).getOrElse(incompatible(aggSig))

  def getType(aggSig: AggSignature): Type = getOption(aggSig).getOrElse(incompatible(aggSig)).out

  def getOption(aggSig: AggSignature): Option[CodeAggregator[T] forSome { type T <: RegionValueAggregator }] =
    getOption(aggSig.op, aggSig.constructorArgs, aggSig.initOpArgs, aggSig.seqOpArgs)

  val getOption: ((AggOp, Seq[Type], Option[Seq[Type]], Seq[Type])) => Option[CodeAggregator[T] forSome { type T <: RegionValueAggregator }] = lift {
    case (Fraction(), Seq(), None, Seq(_: TBoolean)) =>
      CodeAggregator[RegionValueFractionAggregator](TFloat64(), seqOpArgTypes = Array(classOf[Boolean]))

    case (Statistics(), Seq(), None, Seq(_: TFloat64)) =>
      CodeAggregator[RegionValueStatisticsAggregator](RegionValueStatisticsAggregator.typ, seqOpArgTypes = Array(classOf[Double]))

    case (Collect(), Seq(), None, Seq(in)) => in match {
      case _: TBoolean => CodeAggregator[RegionValueCollectBooleanAggregator](TArray(in), seqOpArgTypes = Array(classOf[Boolean]))
      case _: TInt32 | _: TCall => CodeAggregator[RegionValueCollectIntAggregator](TArray(in), seqOpArgTypes = Array(classOf[Int]))
      case _: TInt64 => CodeAggregator[RegionValueCollectLongAggregator](TArray(in), seqOpArgTypes = Array(classOf[Long]))
      case _: TFloat32 => CodeAggregator[RegionValueCollectFloatAggregator](TArray(in), seqOpArgTypes = Array(classOf[Float]))
      case _: TFloat64 => CodeAggregator[RegionValueCollectDoubleAggregator](TArray(in), seqOpArgTypes = Array(classOf[Double]))
      case _ => CodeAggregator[RegionValueCollectAnnotationAggregator](TArray(in), constrArgTypes = Array(classOf[Type]), seqOpArgTypes = Array(classOf[Long]))
    }

    case (InfoScore(), Seq(), None, Seq(TArray(TFloat64(_), _))) =>
      CodeAggregator[RegionValueInfoScoreAggregator](RegionValueInfoScoreAggregator.typ, seqOpArgTypes = Array(classOf[Long]))

    case (Sum(), Seq(), None, Seq(_: TInt64)) => CodeAggregator[RegionValueSumLongAggregator](TInt64(), seqOpArgTypes = Array(classOf[Long]))
    case (Sum(), Seq(), None, Seq(_: TFloat64)) => CodeAggregator[RegionValueSumDoubleAggregator](TFloat64(), seqOpArgTypes = Array(classOf[Double]))
    case (Sum(), Seq(), None, Seq(TArray(TInt64(_), _))) =>
      CodeAggregator[RegionValueArraySumLongAggregator](TArray(TInt64()), seqOpArgTypes = Array(classOf[Long]))

    case (CollectAsSet(), Seq(), None, Seq(in)) =>
      in match {
        case _: TBoolean => CodeAggregator[RegionValueCollectAsSetBooleanAggregator](TSet(TBoolean()), seqOpArgTypes = Array(classOf[Boolean]))
        case _: TInt32 | _: TCall => CodeAggregator[RegionValueCollectAsSetIntAggregator](TSet(in), seqOpArgTypes = Array(classOf[Int]))
        case _: TInt64 => CodeAggregator[RegionValueCollectAsSetLongAggregator](TSet(TInt64()), seqOpArgTypes = Array(classOf[Long]))
        case _: TFloat32 => CodeAggregator[RegionValueCollectAsSetFloatAggregator](TSet(TFloat32()), seqOpArgTypes = Array(classOf[Float]))
        case _: TFloat64 => CodeAggregator[RegionValueCollectAsSetDoubleAggregator](TSet(TFloat64()), seqOpArgTypes = Array(classOf[Double]))
        case _ => CodeAggregator[RegionValueCollectAsSetAnnotationAggregator](TSet(in), constrArgTypes = Array(classOf[Type]), seqOpArgTypes = Array(classOf[Long]))
      }

    case (Sum(), Seq(), None, Seq(TArray(TFloat64(_), _))) =>
      CodeAggregator[RegionValueArraySumDoubleAggregator](TArray(TFloat64()), seqOpArgTypes = Array(classOf[Long]))

    case (Product(), Seq(), None, Seq(_: TInt64)) => CodeAggregator[RegionValueProductLongAggregator](TInt64(), seqOpArgTypes = Array(classOf[Long]))
    case (Product(), Seq(), None, Seq(_: TFloat64)) => CodeAggregator[RegionValueProductDoubleAggregator](TFloat64(), seqOpArgTypes = Array(classOf[Double]))

    case (HardyWeinberg(), Seq(), None, Seq(_: TCall)) => CodeAggregator[RegionValueHardyWeinbergAggregator](
      RegionValueHardyWeinbergAggregator.typ,
      seqOpArgTypes = Array(classOf[Int]))

    case (Max(), Seq(), None, Seq(_: TBoolean)) => CodeAggregator[RegionValueMaxBooleanAggregator](TBoolean(), seqOpArgTypes = Array(classOf[Boolean]))
    case (Max(), Seq(), None, Seq(_: TInt32)) => CodeAggregator[RegionValueMaxIntAggregator](TInt32(), seqOpArgTypes = Array(classOf[Int]))
    case (Max(), Seq(), None, Seq(_: TInt64)) => CodeAggregator[RegionValueMaxLongAggregator](TInt64(), seqOpArgTypes = Array(classOf[Long]))
    case (Max(), Seq(), None, Seq(_: TFloat32)) => CodeAggregator[RegionValueMaxFloatAggregator](TFloat32(), seqOpArgTypes = Array(classOf[Float]))
    case (Max(), Seq(), None, Seq(_: TFloat64)) => CodeAggregator[RegionValueMaxDoubleAggregator](TFloat64(), seqOpArgTypes = Array(classOf[Double]))

    case (Min(), Seq(), None, Seq(_: TBoolean)) => CodeAggregator[RegionValueMinBooleanAggregator](TBoolean(), seqOpArgTypes = Array(classOf[Boolean]))
    case (Min(), Seq(), None, Seq(_: TInt32)) => CodeAggregator[RegionValueMinIntAggregator](TInt32(), seqOpArgTypes = Array(classOf[Int]))
    case (Min(), Seq(), None, Seq(_: TInt64)) => CodeAggregator[RegionValueMinLongAggregator](TInt64(), seqOpArgTypes = Array(classOf[Long]))
    case (Min(), Seq(), None, Seq(_: TFloat32)) => CodeAggregator[RegionValueMinFloatAggregator](TFloat32(), seqOpArgTypes = Array(classOf[Float]))
    case (Min(), Seq(), None, Seq(_: TFloat64)) => CodeAggregator[RegionValueMinDoubleAggregator](TFloat64(), seqOpArgTypes = Array(classOf[Double]))

    case (Count(), Seq(), None, Seq()) => CodeAggregator[RegionValueCountAggregator](TInt64(), seqOpArgTypes = Array())

    case (Counter(), Seq(), None, Seq(in)) => in match {
      case _: TBoolean => CodeAggregator[RegionValueCounterBooleanAggregator](TDict(in, TInt64()), seqOpArgTypes = Array(classOf[Boolean]))
      case _: TInt32 | _: TCall => CodeAggregator[RegionValueCounterIntAggregator](TDict(in, TInt64()), constrArgTypes = Array(classOf[Type]), seqOpArgTypes = Array(classOf[Int]))
      case _: TInt64 => CodeAggregator[RegionValueCounterLongAggregator](TDict(in, TInt64()), constrArgTypes = Array(classOf[Type]), seqOpArgTypes = Array(classOf[Long]))
      case _: TFloat32 => CodeAggregator[RegionValueCounterFloatAggregator](TDict(in, TInt64()), constrArgTypes = Array(classOf[Type]), seqOpArgTypes = Array(classOf[Float]))
      case _: TFloat64 => CodeAggregator[RegionValueCounterDoubleAggregator](TDict(in, TInt64()), constrArgTypes = Array(classOf[Type]), seqOpArgTypes = Array(classOf[Double]))
      case _ => CodeAggregator[RegionValueCounterAnnotationAggregator](TDict(in, TInt64()), constrArgTypes = Array(classOf[Type]), seqOpArgTypes = Array(classOf[Long]))
    }

    case (Take(), constArgs@Seq(_: TInt32), None, Seq(in)) => in match {
      case _: TBoolean => CodeAggregator[RegionValueTakeBooleanAggregator](TArray(in), constrArgTypes = Array(classOf[Int]), seqOpArgTypes = Array(classOf[Boolean]))
      case _: TInt32 | _: TCall => CodeAggregator[RegionValueTakeIntAggregator](TArray(in), constrArgTypes = Array(classOf[Int]), seqOpArgTypes = Array(classOf[Int]))
      case _: TInt64 => CodeAggregator[RegionValueTakeLongAggregator](TArray(in), constrArgTypes = Array(classOf[Int]), seqOpArgTypes = Array(classOf[Long]))
      case _: TFloat32 => CodeAggregator[RegionValueTakeFloatAggregator](TArray(in), constrArgTypes = Array(classOf[Int]), seqOpArgTypes = Array(classOf[Float]))
      case _: TFloat64 => CodeAggregator[RegionValueTakeDoubleAggregator](TArray(in), constrArgTypes = Array(classOf[Int]), seqOpArgTypes = Array(classOf[Double]))
      case _ => CodeAggregator[RegionValueTakeAnnotationAggregator](TArray(in), constrArgTypes = Array(classOf[Int], classOf[Type]), seqOpArgTypes = Array(classOf[Long]))
    }

    case (TakeBy(), constArgs@Seq(_: TInt32), None, Seq(in, key)) =>
      def tbCodeAgg[T <: RegionValueTakeByAggregator](seqOpArgs: Class[_]*)(implicit ct: ClassTag[T]) =
        new CodeAggregator[T](TArray(in), constrArgTypes = Array(classOf[Int], classOf[Type], classOf[Type]), seqOpArgTypes = seqOpArgs.toArray)
      
      (in, key) match {
        case (_: TBoolean, _: TBoolean) => tbCodeAgg[RegionValueTakeByBooleanBooleanAggregator](classOf[Boolean], classOf[Boolean])
        case (_: TBoolean, _: TInt32 | _: TCall) => tbCodeAgg[RegionValueTakeByBooleanIntAggregator](classOf[Boolean], classOf[Int])
        case (_: TBoolean, _: TInt64) => tbCodeAgg[RegionValueTakeByBooleanLongAggregator](classOf[Boolean], classOf[Long])
        case (_: TBoolean, _: TFloat32) => tbCodeAgg[RegionValueTakeByBooleanFloatAggregator](classOf[Boolean], classOf[Float])
        case (_: TBoolean, _: TFloat64) => tbCodeAgg[RegionValueTakeByBooleanDoubleAggregator](classOf[Boolean], classOf[Double])
        case (_: TBoolean, _) => tbCodeAgg[RegionValueTakeByBooleanAnnotationAggregator](classOf[Boolean], classOf[Long])

        case (_: TInt32 | _: TCall, _: TBoolean) => tbCodeAgg[RegionValueTakeByIntBooleanAggregator](classOf[Int], classOf[Boolean])
        case (_: TInt32 | _: TCall, _: TInt32 | _: TCall) => tbCodeAgg[RegionValueTakeByIntIntAggregator](classOf[Int], classOf[Int])
        case (_: TInt32 | _: TCall, _: TInt64) => tbCodeAgg[RegionValueTakeByIntLongAggregator](classOf[Int], classOf[Long])
        case (_: TInt32 | _: TCall, _: TFloat32) => tbCodeAgg[RegionValueTakeByIntFloatAggregator](classOf[Int], classOf[Float])
        case (_: TInt32 | _: TCall, _: TFloat64) => tbCodeAgg[RegionValueTakeByIntDoubleAggregator](classOf[Int], classOf[Double])
        case (_: TInt32 | _: TCall, _) => tbCodeAgg[RegionValueTakeByIntAnnotationAggregator](classOf[Int], classOf[Long])

        case (_: TInt64, _: TBoolean) => tbCodeAgg[RegionValueTakeByLongBooleanAggregator](classOf[Long], classOf[Boolean])
        case (_: TInt64, _: TInt32 | _: TCall) => tbCodeAgg[RegionValueTakeByLongIntAggregator](classOf[Long], classOf[Int])
        case (_: TInt64, _: TInt64) => tbCodeAgg[RegionValueTakeByLongLongAggregator](classOf[Long], classOf[Long])
        case (_: TInt64, _: TFloat32) => tbCodeAgg[RegionValueTakeByLongFloatAggregator](classOf[Long], classOf[Float])
        case (_: TInt64, _: TFloat64) => tbCodeAgg[RegionValueTakeByLongDoubleAggregator](classOf[Long], classOf[Double])
        case (_: TInt64, _) => tbCodeAgg[RegionValueTakeByLongAnnotationAggregator](classOf[Long], classOf[Long])

        case (_: TFloat32, _: TBoolean) => tbCodeAgg[RegionValueTakeByFloatBooleanAggregator](classOf[Float], classOf[Boolean])
        case (_: TFloat32, _: TInt32 | _: TCall) => tbCodeAgg[RegionValueTakeByFloatIntAggregator](classOf[Float], classOf[Int])
        case (_: TFloat32, _: TInt64) => tbCodeAgg[RegionValueTakeByFloatLongAggregator](classOf[Float], classOf[Long])
        case (_: TFloat32, _: TFloat32) => tbCodeAgg[RegionValueTakeByFloatFloatAggregator](classOf[Float], classOf[Float])
        case (_: TFloat32, _: TFloat64) => tbCodeAgg[RegionValueTakeByFloatDoubleAggregator](classOf[Float], classOf[Double])
        case (_: TFloat32, _) => tbCodeAgg[RegionValueTakeByFloatAnnotationAggregator](classOf[Float], classOf[Long])

        case (_: TFloat64, _: TBoolean) => tbCodeAgg[RegionValueTakeByDoubleBooleanAggregator](classOf[Double], classOf[Boolean])
        case (_: TFloat64, _: TInt32 | _: TCall) => tbCodeAgg[RegionValueTakeByDoubleIntAggregator](classOf[Double], classOf[Int])
        case (_: TFloat64, _: TInt64) => tbCodeAgg[RegionValueTakeByDoubleLongAggregator](classOf[Double], classOf[Long])
        case (_: TFloat64, _: TFloat32) => tbCodeAgg[RegionValueTakeByDoubleFloatAggregator](classOf[Double], classOf[Float])
        case (_: TFloat64, _: TFloat64) => tbCodeAgg[RegionValueTakeByDoubleDoubleAggregator](classOf[Double], classOf[Double])
        case (_: TFloat64, _) => tbCodeAgg[RegionValueTakeByDoubleAnnotationAggregator](classOf[Double], classOf[Long])

        case (_, _: TBoolean) => tbCodeAgg[RegionValueTakeByAnnotationBooleanAggregator](classOf[Long], classOf[Boolean])
        case (_, _: TInt32 | _: TCall) => tbCodeAgg[RegionValueTakeByAnnotationIntAggregator](classOf[Long], classOf[Int])
        case (_, _: TInt64) => tbCodeAgg[RegionValueTakeByAnnotationLongAggregator](classOf[Long], classOf[Long])
        case (_, _: TFloat32) => tbCodeAgg[RegionValueTakeByAnnotationFloatAggregator](classOf[Long], classOf[Float])
        case (_, _: TFloat64) => tbCodeAgg[RegionValueTakeByAnnotationDoubleAggregator](classOf[Long], classOf[Double])
        case (_, _) => tbCodeAgg[RegionValueTakeByAnnotationAnnotationAggregator](classOf[Long], classOf[Long])
      }

    case (Histogram(), constArgs@Seq(_: TFloat64, _: TFloat64, _: TInt32), None, Seq(_: TFloat64)) =>
      CodeAggregator[RegionValueHistogramAggregator](
        RegionValueHistogramAggregator.typ,
        constrArgTypes = Array(classOf[Double], classOf[Double], classOf[Int]),
        seqOpArgTypes = Array(classOf[Double]))

    case (CallStats(), Seq(), initOpArgs@Some(Seq(_: TInt32)), Seq(_: TCall)) =>
      CodeAggregator[RegionValueCallStatsAggregator](
        RegionValueCallStatsAggregator.typ,
        initOpArgTypes = Some(Array(classOf[Int])),
        seqOpArgTypes = Array(classOf[Int]))

    case (Inbreeding(), Seq(), None, seqOpArgs@Seq(_: TCall, _: TFloat64)) =>
      CodeAggregator[RegionValueInbreedingAggregator](
        RegionValueInbreedingAggregator.typ,
        seqOpArgTypes = Array(classOf[Int], classOf[Double]))
  }

  private def incompatible(aggSig: AggSignature): Nothing = {
    throw new RuntimeException(s"no aggregator named ${ aggSig.op } taking constructor arguments [${ aggSig.constructorArgs.mkString(", ") }] " +
      s"initOp arguments ${ if (aggSig.initOpArgs.isEmpty) "None" else "[" + aggSig.initOpArgs.get.mkString(", ") + "]" } " +
      s"and seqOp arguments [${ aggSig.seqOpArgs.mkString(", ")}]")
  }

  val fromString: PartialFunction[String, AggOp] = {
    case "fraction" | "Fraction" => Fraction()
    case "stats" | "Statistics" => Statistics()
    case "collect" | "Collect" => Collect()
    case "collectAsSet" | "CollectAsSet" => CollectAsSet()
    case "sum" | "Sum" => Sum()
    case "product" | "Product" => Product()
    case "max" | "Max" => Max()
    case "min" | "Min" => Min()
    case "count" | "Count" => Count()
    case "counter" | "Counter" => Counter()
    case "take" | "Take" => Take()
    case "takeBy" | "TakeBy" => TakeBy()
    case "hist" | "Histogram" => Histogram()
    case "infoScore" | "InfoScore" => InfoScore()
    case "callStats" | "CallStats" => CallStats()
    case "inbreeding" | "Inbreeding" => Inbreeding()
    case "hardyWeinberg" | "HardyWeinberg" => HardyWeinberg()
  }
}

package is.hail.expr.ir

import is.hail.annotations.aggregators._
import is.hail.asm4s._
import is.hail.expr._
import is.hail.utils._

import scala.reflect.ClassTag

object AggOp {
  private val ternary: PartialFunction[(AggOp, Type, Type), TernaryAggregatorCode[_, _, _, _ <: RegionValueAggregator, _]] = {
    case (Histogram(), _: TFloat64, TArray(TFloat64(_), _)) =>
      ternaryAggregatorCode[Double, Double, Int, Double](new RegionValueHistogramAggregator(_, _, _))
  }

  private val nullary: PartialFunction[(AggOp, Type), NullaryAggregatorCode[_ <: RegionValueAggregator, _]] = {
    case (Fraction(), _: TBoolean) => nullaryAggregatorCode[Boolean](new RegionValueFractionAggregator())
    case (Statistics(), _: TFloat64) => nullaryAggregatorCode[Double](new RegionValueStatisticsAggregator())
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


  sealed trait NullaryAggregatorCodeCurried[T] {
    def apply[Agg >: Null : ClassTag : TypeInfo]
      (aggregator: Agg)
      (implicit tct: ClassTag[T], hrt: HailRep[T]): NullaryAggregatorCode[Agg, T] =
      new NullaryAggregatorCode(hailType[T], aggregator)
  }

  private object nullaryAggregatorCodeCurriedInstance extends NullaryAggregatorCodeCurried[Nothing]

  def nullaryAggregatorCode[T] = nullaryAggregatorCodeCurriedInstance.asInstanceOf[NullaryAggregatorCodeCurried[T]]

  class NullaryAggregatorCode[Agg >: Null : ClassTag : TypeInfo, T : ClassTag]
    (t: Type, val aggregator: Agg) {
    def seqOp(rva: Code[RegionValueAggregator], v: Code[T], mv: Code[Boolean]): Code[Unit] =
      mv.mux(
        Code.checkcast[Agg](rva).invoke[T, Boolean, Unit]("seqOp", coerce[T](defaultValue(t)), true),
        Code.checkcast[Agg](rva).invoke[T, Boolean, Unit]("seqOp", coerce[T](v), false))
  }

  sealed trait UnaryAggregatorCodeCurried[T, U] {
    def apply[Agg >: Null : ClassTag : TypeInfo]
      (aggregator: (T) => Agg)
      (implicit uct: ClassTag[U], hrt: HailRep[U]): UnaryAggregatorCode[T, Agg, U] =
      new UnaryAggregatorCode(hailType[U], aggregator)
  }

  private object unaryAggregatorCodeCurriedInstance extends UnaryAggregatorCodeCurried[Nothing, Nothing]

  def unaryAggregatorCode[T, U] = unaryAggregatorCodeCurriedInstance.asInstanceOf[UnaryAggregatorCodeCurried[T, U]]

  class UnaryAggregatorCode[T, Agg >: Null : ClassTag : TypeInfo, U : ClassTag]
    (t: Type, val aggregator: (T) => Agg) {
    def seqOp(rva: Code[RegionValueAggregator], v: Code[U], mv: Code[Boolean]): Code[Unit] =
      mv.mux(
        Code.checkcast[Agg](rva).invoke[U, Boolean, Unit]("seqOp", coerce[U](defaultValue(t)), true),
        Code.checkcast[Agg](rva).invoke[U, Boolean, Unit]("seqOp", coerce[U](v), false))
  }

  sealed trait BinaryAggregatorCodeCurried[T, U, V] {
    def apply[Agg >: Null : ClassTag : TypeInfo]
      (aggregator: (T, U) => Agg)
      (implicit uct: ClassTag[V], hrt: HailRep[V]): BinaryAggregatorCode[T, U, Agg, V] =
      new BinaryAggregatorCode(hailType[V], aggregator)
  }

  private object binaryAggregatorCodeCurriedInstance extends BinaryAggregatorCodeCurried[Nothing, Nothing, Nothing]

  def binaryAggregatorCode[T, U, V] = binaryAggregatorCodeCurriedInstance.asInstanceOf[BinaryAggregatorCodeCurried[T, U, V]]

  class BinaryAggregatorCode[T, U, Agg >: Null : ClassTag : TypeInfo, V : ClassTag]
    (t: Type, val aggregator: (T, U) => Agg) {
    def seqOp(rva: Code[RegionValueAggregator], v: Code[V], mv: Code[Boolean]): Code[Unit] =
      mv.mux(
        Code.checkcast[Agg](rva).invoke[V, Boolean, Unit]("seqOp", coerce[V](defaultValue(t)), true),
        Code.checkcast[Agg](rva).invoke[V, Boolean, Unit]("seqOp", coerce[V](v), false))
  }

  sealed trait TernaryAggregatorCodeCurried[T, U, V, W] {
    def apply[Agg >: Null : ClassTag : TypeInfo]
      (aggregator: (T, U, V) => Agg)
      (implicit uct: ClassTag[W], hrt: HailRep[W]): TernaryAggregatorCode[T, U, V, Agg, W] =
      new TernaryAggregatorCode(hailType[W], aggregator)
  }

  private object ternaryAggregatorCodeCurriedInstance extends TernaryAggregatorCodeCurried[Nothing, Nothing, Nothing, Nothing]

  def ternaryAggregatorCode[T, U, V, W] = ternaryAggregatorCodeCurriedInstance.asInstanceOf[TernaryAggregatorCodeCurried[T, U, V, W]]

  class TernaryAggregatorCode[T, U, V, Agg >: Null : ClassTag : TypeInfo, W : ClassTag]
    (t: Type, val aggregator: (T, U, V) => Agg) {
    def seqOp(rva: Code[RegionValueAggregator], v: Code[W], mv: Code[Boolean]): Code[Unit] =
      mv.mux(
        Code.checkcast[Agg](rva).invoke[W, Boolean, Unit]("seqOp", coerce[W](defaultValue(t)), true),
        Code.checkcast[Agg](rva).invoke[W, Boolean, Unit]("seqOp", coerce[W](v), false))
  }
}

sealed trait AggOp { }
final case class Fraction() extends AggOp { } // remove when prefixes work
final case class Statistics() extends AggOp { } // remove when prefixes work
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
// Fraction === map(x => p(x)).fraction
// Statistics === map(x => Cast(x, TDouble())).stats
// Histogram === map(x => Cast(x, TDouble())).hist

// Counter needs Dict


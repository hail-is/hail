package is.hail.expr.ir

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr._

import scala.reflect.ClassTag

object CodeAggregator {
  type Nullary = NullaryCodeAggregator[_ <: RegionValueAggregator, _]

  def nullary[T] = nullaryCodeAggregatorCurriedInstance.asInstanceOf[NullaryCodeAggregatorCurried[T]]

  type Unary = UnaryCodeAggregator[_, _ <: RegionValueAggregator, _]

  def unary[T, U] = unaryCodeAggregatorCurriedInstance.asInstanceOf[UnaryCodeAggregatorCurried[T, U]]

  type Binary = BinaryCodeAggregator[_, _, _ <: RegionValueAggregator, _]

  def binary[T, U, V] = binaryCodeAggregatorCurriedInstance.asInstanceOf[BinaryCodeAggregatorCurried[T, U, V]]

  type Ternary = TernaryCodeAggregator[_, _, _, _ <: RegionValueAggregator, _]

  def ternary[T, U, V, W] = ternaryCodeAggregatorCurriedInstance.asInstanceOf[TernaryCodeAggregatorCurried[T, U, V, W]]
}

/**
  * Pair the aggregator with a staged seqOp that calls the non-generic seqOp
  * method and handles missingness correctly
  *
  **/
class NullaryCodeAggregator[Agg <: RegionValueAggregator : ClassTag : TypeInfo, T : ClassTag]
  (t: Type, val aggregator: Agg) {
  def seqOp(rva: Code[RegionValueAggregator], v: Code[_], mv: Code[Boolean]): Code[Unit] =
    mv.mux(
      Code.checkcast[Agg](rva).invoke[T, Boolean, Unit]("seqOp", coerce[T](defaultValue(t)), true),
      Code.checkcast[Agg](rva).invoke[T, Boolean, Unit]("seqOp", coerce[T](v), false))
}

class UnaryCodeAggregator[T, Agg <: RegionValueAggregator : ClassTag : TypeInfo, U : ClassTag]
  (t: Type, val aggregator: (T) => Agg) {
  def seqOp(rva: Code[RegionValueAggregator], v: Code[_], mv: Code[Boolean]): Code[Unit] =
    mv.mux(
      Code.checkcast[Agg](rva).invoke[U, Boolean, Unit]("seqOp", coerce[U](defaultValue(t)), true),
      Code.checkcast[Agg](rva).invoke[U, Boolean, Unit]("seqOp", coerce[U](v), false))
}

class BinaryCodeAggregator[T, U, Agg <: RegionValueAggregator : ClassTag : TypeInfo, V : ClassTag]
  (t: Type, val aggregator: (T, U) => Agg) {
  def seqOp(rva: Code[RegionValueAggregator], v: Code[_], mv: Code[Boolean]): Code[Unit] =
    mv.mux(
      Code.checkcast[Agg](rva).invoke[V, Boolean, Unit]("seqOp", coerce[V](defaultValue(t)), true),
      Code.checkcast[Agg](rva).invoke[V, Boolean, Unit]("seqOp", coerce[V](v), false))
}

class TernaryCodeAggregator[T, U, V, Agg <: RegionValueAggregator : ClassTag : TypeInfo, W : ClassTag]
  (t: Type, val aggregator: (Code[T], Code[Boolean], Code[U], Code[Boolean], Code[V], Code[Boolean]) => Code[Agg]) {
  def seqOp(rva: Code[RegionValueAggregator], v: Code[_], mv: Code[Boolean]): Code[Unit] =
    mv.mux(
      Code.checkcast[Agg](rva).invoke[W, Boolean, Unit]("seqOp", coerce[W](defaultValue(t)), true),
      Code.checkcast[Agg](rva).invoke[W, Boolean, Unit]("seqOp", coerce[W](v), false))
}


/**
  * Curries the type arguments which enables inference on Agg, with manual
  * annotation of T, U, V, and W
  *
  **/
sealed trait NullaryCodeAggregatorCurried[T] {
  def apply[Agg <: RegionValueAggregator : ClassTag : TypeInfo]
    (aggregator: Agg)
    (implicit tct: ClassTag[T], hrt: HailRep[T]): NullaryCodeAggregator[Agg, T] =
    new NullaryCodeAggregator(hailType[T], aggregator)
}

private object nullaryCodeAggregatorCurriedInstance extends NullaryCodeAggregatorCurried[Nothing]

sealed trait UnaryCodeAggregatorCurried[T, U] {
  def apply[Agg <: RegionValueAggregator : ClassTag : TypeInfo]
    (aggregator: (T) => Agg)
    (implicit uct: ClassTag[U], hrt: HailRep[U]): UnaryCodeAggregator[T, Agg, U] =
    new UnaryCodeAggregator(hailType[U], aggregator)
}

private object unaryCodeAggregatorCurriedInstance extends UnaryCodeAggregatorCurried[Nothing, Nothing]

sealed trait BinaryCodeAggregatorCurried[T, U, V] {
  def apply[Agg <: RegionValueAggregator : ClassTag : TypeInfo]
    (aggregator: (T, U) => Agg)
    (implicit uct: ClassTag[V], hrt: HailRep[V]): BinaryCodeAggregator[T, U, Agg, V] =
    new BinaryCodeAggregator(hailType[V], aggregator)
}

private object binaryCodeAggregatorCurriedInstance extends BinaryCodeAggregatorCurried[Nothing, Nothing, Nothing]

sealed trait TernaryCodeAggregatorCurried[T, U, V, W] {
  def apply[Agg <: RegionValueAggregator : ClassTag : TypeInfo]
    (aggregator: (Code[T], Code[Boolean], Code[U], Code[Boolean], Code[V], Code[Boolean]) => Code[Agg])
    (implicit uct: ClassTag[W], hrt: HailRep[W]): TernaryCodeAggregator[T, U, V, Agg, W] =
    new TernaryCodeAggregator(hailType[W], aggregator)
}

private object ternaryCodeAggregatorCurriedInstance extends TernaryCodeAggregatorCurried[Nothing, Nothing, Nothing, Nothing]

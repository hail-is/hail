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
  (in: Type, val aggregator: Agg, val out: Type) {
  def seqOp(rva: Code[RegionValueAggregator], v: Code[_], mv: Code[Boolean]): Code[Unit] = {
    mv.mux(
      Code.checkcast[Agg](rva).invoke[T, Boolean, Unit]("seqOp", coerce[T](defaultValue(in)), true),
      Code.checkcast[Agg](rva).invoke[T, Boolean, Unit]("seqOp", coerce[T](v), false))
  }
}

class UnaryCodeAggregator[T, Agg <: RegionValueAggregator : ClassTag : TypeInfo, U : ClassTag]
  (in: Type, val aggregator: (Code[T], Code[Boolean]) => Code[Agg], val out: Type) {
  def seqOp(rva: Code[RegionValueAggregator], v: Code[_], mv: Code[Boolean]): Code[Unit] =
    mv.mux(
      Code.checkcast[Agg](rva).invoke[U, Boolean, Unit]("seqOp", coerce[U](defaultValue(in)), true),
      Code.checkcast[Agg](rva).invoke[U, Boolean, Unit]("seqOp", coerce[U](v), false))
}

class BinaryCodeAggregator[T, U, Agg <: RegionValueAggregator : ClassTag : TypeInfo, V : ClassTag]
  (in: Type, val aggregator: (Code[T], Code[Boolean], Code[U], Code[Boolean]) => Code[Agg], val out: Type) {
  def seqOp(rva: Code[RegionValueAggregator], v: Code[_], mv: Code[Boolean]): Code[Unit] =
    mv.mux(
      Code.checkcast[Agg](rva).invoke[V, Boolean, Unit]("seqOp", coerce[V](defaultValue(in)), true),
      Code.checkcast[Agg](rva).invoke[V, Boolean, Unit]("seqOp", coerce[V](v), false))
}

class TernaryCodeAggregator[T, U, V, Agg <: RegionValueAggregator : ClassTag : TypeInfo, W : ClassTag]
  (in: Type, val aggregator: (Code[T], Code[Boolean], Code[U], Code[Boolean], Code[V], Code[Boolean]) => Code[Agg], val out: Type) {
  def seqOp(rva: Code[RegionValueAggregator], v: Code[_], mv: Code[Boolean]): Code[Unit] =
    mv.mux(
      Code.checkcast[Agg](rva).invoke[W, Boolean, Unit]("seqOp", coerce[W](defaultValue(in)), true),
      Code.checkcast[Agg](rva).invoke[W, Boolean, Unit]("seqOp", coerce[W](v), false))
}


/**
  * Curries the type arguments which enables inference on Agg, with manual
  * annotation of T, U, V, and W
  *
  **/
sealed trait NullaryCodeAggregatorCurried[T] {
  def apply[Agg <: RegionValueAggregator : ClassTag : TypeInfo]
    (aggregator: Agg, out: Type)
    (implicit tct: ClassTag[T], hrt: HailRep[T]): NullaryCodeAggregator[Agg, T] =
    new NullaryCodeAggregator(hailType[T], aggregator, out)
}

private object nullaryCodeAggregatorCurriedInstance extends NullaryCodeAggregatorCurried[Nothing]

sealed trait UnaryCodeAggregatorCurried[T, U] {
  def apply[Agg <: RegionValueAggregator : ClassTag : TypeInfo]
    (aggregator: (Code[T], Code[Boolean]) => Code[Agg], out: Type)
    (implicit uct: ClassTag[U], hrt: HailRep[U]): UnaryCodeAggregator[T, Agg, U] =
    new UnaryCodeAggregator(hailType[U], aggregator, out)
}

private object unaryCodeAggregatorCurriedInstance extends UnaryCodeAggregatorCurried[Nothing, Nothing]

sealed trait BinaryCodeAggregatorCurried[T, U, V] {
  def apply[Agg <: RegionValueAggregator : ClassTag : TypeInfo]
    (aggregator: (Code[T], Code[Boolean], Code[U], Code[Boolean]) => Code[Agg], out: Type)
    (implicit uct: ClassTag[V], hrt: HailRep[V]): BinaryCodeAggregator[T, U, Agg, V] =
    new BinaryCodeAggregator(hailType[V], aggregator, out)
}

private object binaryCodeAggregatorCurriedInstance extends BinaryCodeAggregatorCurried[Nothing, Nothing, Nothing]

sealed trait TernaryCodeAggregatorCurried[T, U, V, W] {
  def apply[Agg <: RegionValueAggregator : ClassTag : TypeInfo]
    (aggregator: (Code[T], Code[Boolean], Code[U], Code[Boolean], Code[V], Code[Boolean]) => Code[Agg], out: Type)
    (implicit uct: ClassTag[W], hrt: HailRep[W]): TernaryCodeAggregator[T, U, V, Agg, W] =
    new TernaryCodeAggregator(hailType[W], aggregator, out)
}

private object ternaryCodeAggregatorCurriedInstance extends TernaryCodeAggregatorCurried[Nothing, Nothing, Nothing, Nothing]

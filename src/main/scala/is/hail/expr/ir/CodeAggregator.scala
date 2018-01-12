package is.hail.expr.ir

import is.hail.annotations._
import is.hail.annotations.aggregators._
import is.hail.asm4s._
import is.hail.expr.types._
import is.hail.expr.{HailRep, hailType}

import scala.reflect.ClassTag

object CodeAggregator {
  def apply[T] = codeAggregatorCurriedInstance.asInstanceOf[CodeAggregatorCurried[T]]
}

/**
  * Pair the aggregator with a staged seqOp that calls the non-generic seqOp
  * method and handles missingness correctly
  *
  **/
class CodeAggregator[Agg <: RegionValueAggregator : ClassTag : TypeInfo, T : ClassTag]
  (in: Type, val stagedNew: (Array[Code[_]], Array[Code[Boolean]]) => Code[Agg], val out: Type) {
  def seqOp(rva: Code[RegionValueAggregator], v: Code[_], mv: Code[Boolean]): Code[Unit] = {
    mv.mux(
      Code.checkcast[Agg](rva).invoke[T, Boolean, Unit]("seqOp", coerce[T](defaultValue(in)), true),
      Code.checkcast[Agg](rva).invoke[T, Boolean, Unit]("seqOp", coerce[T](v), false))
  }
}

/**
  * Curries the type arguments which enables inference on Agg, with manual
  * annotation of T
  *
  **/
sealed trait CodeAggregatorCurried[T] {
  def apply[Agg <: RegionValueAggregator : ClassTag : TypeInfo]
    (stagedNew: (Array[Code[_]], Array[Code[Boolean]]) => Code[Agg], out: Type)
    (implicit tct: ClassTag[T], hrt: HailRep[T]): CodeAggregator[Agg, T] =
    new CodeAggregator(hailType[T], stagedNew, out)
}

private object codeAggregatorCurriedInstance extends CodeAggregatorCurried[Nothing]



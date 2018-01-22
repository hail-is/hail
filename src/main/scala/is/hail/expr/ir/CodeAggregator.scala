package is.hail.expr.ir

import is.hail.annotations._
import is.hail.annotations.aggregators._
import is.hail.asm4s._
import is.hail.expr.types._
import is.hail.expr.{HailRep, hailType}

import scala.reflect.ClassTag
import scala.reflect.classTag

/**
  * Pair the aggregator with a staged seqOp that calls the non-generic seqOp
  * method and handles missingness correctly
  *
  **/
case class CodeAggregator[Agg <: RegionValueAggregator : ClassTag : TypeInfo]
  (in: Type, out: Type, constructorArgumentTypes: Class[_]*) {
  def seqOp(rva: Code[RegionValueAggregator], v: Code[_], mv: Code[Boolean]): Code[Unit] = {
    TypeToIRIntermediateClassTag(in) match {
      case ct: ClassTag[t] =>
        mv.mux(
          Code.checkcast[Agg](rva).invoke("seqOp", coerce[t](defaultValue(in)), true)(ct, classTag[Boolean], classTag[Unit]),
          Code.checkcast[Agg](rva).invoke("seqOp", coerce[t](v), false)(ct, classTag[Boolean], classTag[Unit]))
    }
  }

  def stagedNew(v: Array[Code[_]], m: Array[Code[Boolean]]): Code[Agg] = {
    val anyArgMissing = m.fold[Code[Boolean]](false)(_ | _)
    anyArgMissing.mux(
      Code._throw(Code.newInstance[RuntimeException, String]("Aggregators must have non missing arguments")),
      Code.newInstance[Agg](constructorArgumentTypes.toArray, v))
  }
}

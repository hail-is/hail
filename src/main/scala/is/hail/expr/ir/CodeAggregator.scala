package is.hail.expr.ir

import is.hail.annotations.Region
import is.hail.annotations.aggregators._
import is.hail.asm4s._
import is.hail.expr.types._

import scala.reflect.ClassTag
import scala.reflect.classTag

/**
  * Pair the aggregator with a staged seqOp that calls the non-generic seqOp
  * method and handles missingness correctly. If an initOp exists, the arguments
  * have already had missingness handled.
  **/
case class CodeAggregator[Agg <: RegionValueAggregator : ClassTag : TypeInfo](
  in: Type,
  out: Type,
  constrArgTypes: Array[Class[_]] = Array.empty[Class[_]],
  initOpArgTypes: Option[Array[Class[_]]] = None) {

  def initOp(rva: Code[RegionValueAggregator], vs: Array[Code[_]], ms: Array[Code[Boolean]]): Code[Unit] = {
    assert(initOpArgTypes.isDefined && vs.length == ms.length)
    val argTypes = initOpArgTypes.get.flatMap[Class[_], Array[Class[_]]](Array(_, classOf[Boolean]))
    val args = vs.zip(ms).flatMap { case (v, m) => Array(v, m) }
    Code.checkcast[Agg](rva).invoke("initOp", argTypes, args)(classTag[Unit])
  }

  def seqOp(rva: Code[RegionValueAggregator], v: Code[_], mv: Code[Boolean]): Code[Unit] = {
    TypeToIRIntermediateClassTag(in) match {
      case ct: ClassTag[t] =>
        mv.mux(
          Code.checkcast[Agg](rva).invoke("seqOp", coerce[t](defaultValue(in)), true)(ct, classTag[Boolean], classTag[Unit]),
          Code.checkcast[Agg](rva).invoke("seqOp", coerce[t](v), false)(ct, classTag[Boolean], classTag[Unit]))
    }
  }

  def stagedNew(v: Array[Code[_]], m: Array[Code[Boolean]]): Code[Agg] = {
    assert(v.length == m.length)
    val anyArgMissing = m.fold[Code[Boolean]](false)(_ | _)
    anyArgMissing.mux(
      Code._throw(Code.newInstance[RuntimeException, String]("Aggregators must have non missing constructor arguments")),
      Code.newInstance[Agg](constrArgTypes, v))
  }
}

package is.hail.expr.ir

import is.hail.annotations.{Region, UnsafeRow}
import is.hail.annotations.aggregators._
import is.hail.asm4s._
import is.hail.expr.types._

import scala.collection.mutable
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

  def seqOp(region: Code[Region], rva: Code[RegionValueAggregator], v: Code[_], mv: Code[Boolean]): Code[Unit] = {
    TypeToIRIntermediateClassTag(in) match {
      case ct: ClassTag[t] =>
        mv.mux(
          Code.checkcast[Agg](rva).invoke(
            "seqOp",
            region,
            coerce[t](defaultValue(in)),
            true
          )(classTag[Region],
            ct,
            classTag[Boolean],
            classTag[Unit]
          ),
          Code.checkcast[Agg](rva).invoke(
            "seqOp",
            region,
            coerce[t](v),
            false
          )(classTag[Region],
            ct,
            classTag[Boolean],
            classTag[Unit]
          ))
    }
  }

  def toKeyedAggregator(keyType: Type): KeyedCodeAggregator[Agg] =
    KeyedCodeAggregator[Agg](keyType, in, out, initOpArgTypes)

  def stagedNew(v: Array[Code[_]], m: Array[Code[Boolean]]): Code[Agg] = {
    assert(v.length == m.length)
    val anyArgMissing = m.fold[Code[Boolean]](false)(_ | _)
    anyArgMissing.mux(
      Code._throw(Code.newInstance[RuntimeException, String]("Aggregators must have non missing constructor arguments")),
      Code.newInstance[Agg](constrArgTypes, v))
  }
}

case class KeyedCodeAggregator[Agg <: RegionValueAggregator : ClassTag : TypeInfo](keyType: Type, inType: Type, outType: Type,
                               initOpArgTypes: Option[Array[Class[_]]] = None) {

  def initOp(krva: Code[RegionValueAggregator], vs: Array[Code[_]], ms: Array[Code[Boolean]]): Code[Unit] = {
    assert(initOpArgTypes.isDefined && vs.length == ms.length)
    val argTypes = initOpArgTypes.get.flatMap[Class[_], Array[Class[_]]](Array(_, classOf[Boolean]))
    val args = vs.zip(ms).flatMap { case (v, m) => Array(v, m) }
    val krvAgg = Code.checkcast[KeyedRegionValueAggregator[Agg]](krva)
    krvAgg.invoke[Agg]("rvAgg").invoke("initOp", argTypes, args)(classTag[Unit])
  }

  def seqOp(region: Code[Region], krva: Code[RegionValueAggregator], k: Code[_], mk: Code[Boolean], v: Code[_], mv: Code[Boolean]): Code[Unit] = {

    val krvAgg = Code.checkcast[KeyedRegionValueAggregator[Agg]](krva)

    def wrapArg(arg: Code[_]): Code[_] = keyType match {
      case _: TBoolean => Code.boxBoolean(coerce[Boolean](arg))
      case _: TInt32 => Code.boxInt(coerce[Int](arg))
      case _: TInt64 => Code.boxLong(coerce[Long](arg))
      case _: TFloat32 => Code.boxFloat(coerce[Float](arg))
      case _: TFloat64 => Code.boxDouble(coerce[Double](arg))
      case _: TString =>
        Code.invokeScalaObject[Region, Long, String](
          TString.getClass, "loadString",
          region, coerce[Long](arg))
      case _ =>
        Code.invokeScalaObject[Type, Region, Long, Any](
          UnsafeRow.getClass, "read",
          krvAgg.invoke[Type]("keyType"), region, coerce[Long](arg))
    }

    val key = mk.mux(Code._null[Any], wrapArg(k))
    val m =  krvAgg.invoke[mutable.Map[Any, Agg]]("m")

    val rva = Code(m.invoke[Any, Boolean]("contains", key).mux(
        Code._empty,
        m.invoke[Any, Any, Unit]("update", key, Code.checkcast[Agg](krvAgg.invoke[RegionValueAggregator]("rvAgg")).invoke[Agg]("copy"))),
      m.invoke("apply", key))

    TypeToIRIntermediateClassTag(inType) match {
      case ct: ClassTag[t] =>
        mv.mux(
          Code.checkcast[Agg](rva).invoke(
            "seqOp",
            region,
            coerce[t](defaultValue(inType)),
            true
          )(classTag[Region],
            ct,
            classTag[Boolean],
            classTag[Unit]
          ),
          Code.checkcast[Agg](rva).invoke(
            "seqOp",
            region,
            coerce[t](v),
            false
          )(classTag[Region],
            ct,
            classTag[Boolean],
            classTag[Unit]
          ))
    }
  }
}

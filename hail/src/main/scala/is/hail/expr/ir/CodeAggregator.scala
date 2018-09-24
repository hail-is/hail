package is.hail.expr.ir

import is.hail.annotations.{Region, SafeRow}
import is.hail.annotations.aggregators.{KeyedRegionValueAggregator, _}
import is.hail.asm4s._
import is.hail.expr.types._
import is.hail.expr.types.physical.PType

import scala.reflect.ClassTag
import scala.reflect.classTag

abstract class BaseCodeAggregator[Agg <: RegionValueAggregator : ClassTag : TypeInfo] {
  def out: Type

  def initOpArgTypes: Option[Array[Class[_]]]

  def seqOpArgTypes: Array[Class[_]]

  def initOp(mb: EmitMethodBuilder, rva: Code[RegionValueAggregator], vs: Array[Code[_]], ms: Array[Code[Boolean]]): Code[Unit]

  def seqOp(mb: EmitMethodBuilder, region: Code[Region], rva: Code[RegionValueAggregator], vs: Array[Code[_]], ms: Array[Code[Boolean]]): Code[Unit]

  def toKeyedAggregator(keyType: Type): KeyedCodeAggregator[Agg]
}

/**
  * Pair the aggregator with a staged seqOp that calls the non-generic seqOp and initOp
  * methods. Missingness is handled by Emit.
  **/
case class CodeAggregator[Agg <: RegionValueAggregator : ClassTag : TypeInfo](
  out: Type,
  constrArgTypes: Array[Class[_]] = Array.empty[Class[_]],
  initOpArgTypes: Option[Array[Class[_]]] = None,
  seqOpArgTypes: Array[Class[_]] = Array.empty[Class[_]]) extends BaseCodeAggregator[Agg] {

  def initOp(mb: EmitMethodBuilder, rva: Code[RegionValueAggregator], vs: Array[Code[_]], ms: Array[Code[Boolean]]): Code[Unit] = {
    assert(initOpArgTypes.isDefined && vs.length == ms.length)
    val argTypes = initOpArgTypes.get.flatMap(Array[Class[_]](_, classOf[Boolean]))
    val args = vs.zip(ms).flatMap { case (v, m) => Array(v, m) }
    Code.checkcast[Agg](rva).invoke("initOp", argTypes, args)(classTag[Unit])
  }

  def seqOp(mb: EmitMethodBuilder, region: Code[Region], rva: Code[RegionValueAggregator], vs: Array[Code[_]], ms: Array[Code[Boolean]]): Code[Unit] = {
    assert(vs.length == ms.length)
    val argTypes = seqOpArgTypes.flatMap(Array[Class[_]](_, classOf[Boolean]))
    val args = vs.zip(ms).flatMap { case (v, m) => Array(v, m) }
    Code.checkcast[Agg](rva).invoke("seqOp", Array(classOf[Region]) ++ argTypes, Array(region) ++ args)(classTag[Unit])
  }

  def stagedNew(v: Array[Code[_]], m: Array[Code[Boolean]]): Code[Agg] = {
    assert(v.length == m.length)
    val anyArgMissing = m.fold[Code[Boolean]](false)(_ | _)
    anyArgMissing.mux(
      Code._throw(Code.newInstance[RuntimeException, String]("Aggregators must have non missing constructor arguments")),
      Code.newInstance[Agg](constrArgTypes, v))
  }

  def toKeyedAggregator(keyType: Type): KeyedCodeAggregator[Agg] =
    KeyedCodeAggregator[Agg](Array(keyType), TDict(keyType, out), initOpArgTypes, seqOpArgTypes)
}

case class KeyedCodeAggregator[Agg <: RegionValueAggregator : ClassTag : TypeInfo](
  keys: Array[Type],
  out: Type,
  initOpArgTypes: Option[Array[Class[_]]] = None,
  seqOpArgTypes: Array[Class[_]] = Array.empty[Class[_]]) extends BaseCodeAggregator[Agg] {

  def getRVAgg(mb: EmitMethodBuilder, krvAgg: Code[KeyedRegionValueAggregator], nKeys: Int): Code[RegionValueAggregator] = {
    val krva = mb.newLocal[KeyedRegionValueAggregator]
    val rvAgg = mb.newLocal[RegionValueAggregator]

    val code = Code(
      krva := krvAgg,
      rvAgg := krva.invoke[RegionValueAggregator]("rvAgg"),
      rvAgg
    )

    if (nKeys == 1)
      code
    else
      getRVAgg(mb, Code.checkcast[KeyedRegionValueAggregator](code), nKeys - 1)
  }

  def initOp(mb: EmitMethodBuilder, krva: Code[RegionValueAggregator], vs: Array[Code[_]], ms: Array[Code[Boolean]]): Code[Unit] = {
    assert(initOpArgTypes.isDefined && vs.length == ms.length)
    val argTypes = initOpArgTypes.get.flatMap(Array[Class[_]](_, classOf[Boolean]))
    val args = vs.zip(ms).flatMap { case (v, m) => Array(v, m) }
    val rvAgg = getRVAgg(mb, Code.checkcast[KeyedRegionValueAggregator](krva), keys.length)
    Code.checkcast[Agg](rvAgg).invoke("initOp", argTypes, args)(classTag[Unit])
  }

  def getRVAggByKey(
    mb: EmitMethodBuilder,
    krvAgg: Code[KeyedRegionValueAggregator],
    wrappedKeys: Array[Code[AnyRef]]): Code[Agg] = {

    val krva = mb.newLocal[KeyedRegionValueAggregator]
    val m = mb.newLocal[java.util.HashMap[AnyRef, RegionValueAggregator]]("m")
    val wrappedKey = mb.newLocal[AnyRef]("wrappedKey")
    val nextrva = mb.newLocal[RegionValueAggregator]

    val setup = Code(
      krva := krvAgg,
      m := krva.invoke[java.util.HashMap[AnyRef, RegionValueAggregator]]("m"),
      wrappedKey := wrappedKeys.head
    )

    val code = Code(
      setup,
      Code.toUnit(m.invoke[AnyRef, Boolean]("containsKey", wrappedKey).mux(
        Code._null,
        m.invoke[AnyRef, AnyRef, AnyRef]("put", wrappedKey, krva.invoke[RegionValueAggregator]("rvAgg").invoke[RegionValueAggregator]("copy")))),
      nextrva := m.invoke[AnyRef, RegionValueAggregator]("get", wrappedKey),
      nextrva
    )

    if (wrappedKeys.length == 1) {
      Code.checkcast[Agg](code)
    } else {
      getRVAggByKey(mb, Code.checkcast[KeyedRegionValueAggregator](code), wrappedKeys.drop(1))
    }
  }

  def seqOp(mb: EmitMethodBuilder, region: Code[Region], krva: Code[RegionValueAggregator], vs: Array[Code[_]], ms: Array[Code[Boolean]]): Code[Unit] = {
    assert(vs.length == ms.length)

    def wrapArg(kType: Type, arg: Code[_]): Code[AnyRef] = kType match {
      case _: TBoolean => Code.boxBoolean(coerce[Boolean](arg))
      case _: TInt32 | _: TCall => Code.boxInt(coerce[Int](arg))
      case _: TInt64 => Code.boxLong(coerce[Long](arg))
      case _: TFloat32 => Code.boxFloat(coerce[Float](arg))
      case _: TFloat64 => Code.boxDouble(coerce[Double](arg))
      case _: TString =>
        Code.invokeScalaObject[Region, Long, String](
          TString.getClass, "loadString",
          region, coerce[Long](arg))
      case _ =>
        Code.invokeScalaObject[PType, Region, Long, AnyRef](
          SafeRow.getClass, "read",
          mb.getPType(kType.physicalType), region, coerce[Long](arg))
    }

    val wrappedKeys = keys.zipWithIndex.map { case (kType, i) => ms(i).mux(Code._null, wrapArg(kType, vs(i))) }
    val krvAgg = Code.checkcast[KeyedRegionValueAggregator](krva)
    val rva = getRVAggByKey(mb, krvAgg, wrappedKeys)

    val nKeys = keys.length
    val argTypes = classOf[Region] +: seqOpArgTypes.flatMap(Array[Class[_]](_, classOf[Boolean]))
    val args = vs.drop(nKeys).zip(ms.drop(nKeys)).flatMap { case (v, m) => Array(v, m) }

    rva.invoke("seqOp", argTypes, Array(region) ++ args)(classTag[Unit])
  }

  def toKeyedAggregator(keyType: Type): KeyedCodeAggregator[Agg] =
    KeyedCodeAggregator[Agg](Array(keyType) ++ keys, TDict(keyType, out), initOpArgTypes, seqOpArgTypes)
}

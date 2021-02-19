package is.hail.expr.ir

import is.hail.HailContext
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.types._
import is.hail.utils._
import is.hail.experimental.ExperimentalFunctions
import is.hail.types.physical._
import is.hail.types.physical.stypes._
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives._
import is.hail.types.physical.stypes.concrete._
import is.hail.types.virtual._
import is.hail.variant.Locus
import org.apache.spark.sql.Row

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.reflect._
import java.util.function.Supplier
import scala.collection.mutable.ArrayBuffer

package object functions {
  def tv(name: String): TVariable =
    TVariable(name)

  def tv(name: String, cond: String): TVariable =
    TVariable(name, cond)

  def tnum(name: String): TVariable =
    tv(name, "numeric")

  def wrapArg(r: EmitRegion, t: PType): Code[_] => Code[_] = t match {
    case _: PBoolean => coerce[Boolean]
    case _: PInt32 => coerce[Int]
    case _: PInt64 => coerce[Long]
    case _: PFloat32 => coerce[Float]
    case _: PFloat64 => coerce[Double]
    case _: PCall => coerce[Int]
    case t: PString => c => t.loadString(coerce[Long](c))
    case t: PLocus => c => EmitCodeBuilder.scopedCode(r.mb)(cb => PCode(t, c).asLocus.getLocusObj(cb))
    case _ => c =>
      Code.invokeScalaObject3[PType, Region, Long, Any](
        UnsafeRow.getClass, "read",
        r.mb.getPType(t),
        r.region, coerce[Long](c))
  }

  def boxedTypeInfo(t: Type): TypeInfo[_ >: Null] = t match {
    case TBoolean => classInfo[java.lang.Boolean]
    case TInt32 => classInfo[java.lang.Integer]
    case TInt64 => classInfo[java.lang.Long]
    case TFloat32 => classInfo[java.lang.Float]
    case TFloat64 => classInfo[java.lang.Double]
    case TCall => classInfo[java.lang.Integer]
    case TString => classInfo[java.lang.String]
    case _: TLocus => classInfo[Locus]
    case _ => classInfo[AnyRef]
  }

  def scodeToJavaValue(cb: EmitCodeBuilder, r: Value[Region], sc: SCode): Code[AnyRef] = {
    sc.st match {
      case _: SInt32 => Code.boxInt(sc.asInt32.intCode(cb))
      case _: SInt64 => Code.boxLong(sc.asInt64.longCode(cb))
      case _: SFloat32 => Code.boxFloat(sc.asFloat32.floatCode(cb))
      case _: SFloat64 => Code.boxDouble(sc.asFloat64.doubleCode(cb))
      case _: SBoolean => Code.boxBoolean(sc.asBoolean.boolCode(cb))
      case _: SCall => Code.boxInt(coerce[Int](sc.asPCode.code))
      case _: SString => sc.asString.loadString()
      case _: SLocus => sc.asLocus.getLocusObj(cb)
      case t =>
        val pt = t.canonicalPType()
        val addr = pt.store(cb, r, sc, deepCopy = false)
        Code.invokeScalaObject3[PType, Region, Long, AnyRef](
          UnsafeRow.getClass, "readAnyRef",
          cb.emb.getPType(pt),
          r, addr)

    }
  }

  def boxArg(r: EmitRegion, t: PType): Code[_] => Code[AnyRef] = t match {
    case _: PBoolean => c => Code.boxBoolean(coerce[Boolean](c))
    case _: PInt32 => c => Code.boxInt(coerce[Int](c))
    case _: PInt64 => c => Code.boxLong(coerce[Long](c))
    case _: PFloat32 => c => Code.boxFloat(coerce[Float](c))
    case _: PFloat64 => c => Code.boxDouble(coerce[Double](c))
    case _: PCall => c => Code.boxInt(coerce[Int](c))
    case t: PString => c => t.loadString(coerce[Long](c))
    case t: PLocus => c => EmitCodeBuilder.scopedCode(r.mb)(cb => PCode(t, c).asLocus.getLocusObj(cb))
    case _ => c =>
      Code.invokeScalaObject3[PType, Region, Long, AnyRef](
        UnsafeRow.getClass, "readAnyRef",
        r.mb.getPType(t),
        r.region, coerce[Long](c))
  }

  def unwrapReturn(cb: EmitCodeBuilder, r: Value[Region], pt: PType, value: Code[_]): PCode = pt.virtualType match {
    case TBoolean => PCode(pt, value)
    case TInt32 => PCode(pt, value)
    case TInt64 => PCode(pt, value)
    case TFloat32 => PCode(pt, value)
    case TFloat64 => PCode(pt, value)
    case TString =>
      val st = SStringPointer(pt.asInstanceOf[PCanonicalString])
      st.constructFromString(cb, r, coerce[String](value))
    case TCall => PCode(pt, value)
    case TArray(TInt32) =>
      val pca = pt.asInstanceOf[PCanonicalArray]
      val arr = cb.newLocal[IndexedSeq[Int]]("unrwrap_return_array_int32_arr", coerce[IndexedSeq[Int]](value))
      val len = cb.newLocal[Int]("unwrap_return_array_int32_len", arr.invoke[Int]("length"))
      pca.constructFromElements(cb, r, len, deepCopy = false) { (cb, idx) =>
        val elt = cb.newLocal[java.lang.Integer]("unwrap_return_array_int32_elt",
          Code.checkcast[java.lang.Integer](arr.invoke[Int, java.lang.Object]("apply", idx)))
        IEmitCode(cb, elt.isNull, primitive(elt.invoke[Int]("intValue")))
      }
    case TArray(TFloat64) =>
      val pca = pt.asInstanceOf[PCanonicalArray]
      val arr = cb.newLocal[IndexedSeq[Double]]("unrwrap_return_array_float64_arr", coerce[IndexedSeq[Double]](value))
      val len = cb.newLocal[Int]("unwrap_return_array_float64_len", arr.invoke[Int]("length"))
      pca.constructFromElements(cb, r, len, deepCopy = false) { (cb, idx) =>
        val elt = cb.newLocal[java.lang.Double]("unwrap_return_array_float64_elt",
          Code.checkcast[java.lang.Double](arr.invoke[Int, java.lang.Object]("apply", idx)))
        IEmitCode(cb, elt.isNull, primitive(elt.invoke[Double]("doubleValue")))
      }
    case TArray(TString) =>
      val pca = pt.asInstanceOf[PCanonicalArray]
      val arr = cb.newLocal[IndexedSeq[String]]("unrwrap_return_array_str_arr", coerce[IndexedSeq[String]](value))
      val len = cb.newLocal[Int]("unwrap_return_array_str_len", arr.invoke[Int]("length"))
      pca.constructFromElements(cb, r, len, deepCopy = false) { (cb, idx) =>
        val st = SStringPointer(pca.elementType.asInstanceOf[PCanonicalString])
        val elt = cb.newLocal[String]("unwrap_return_array_str_elt",
          Code.checkcast[String](arr.invoke[Int, java.lang.Object]("apply", idx)))
        IEmitCode(cb, elt.isNull, st.constructFromString(cb, r, elt))
      }
    case t: TBaseStruct =>
      val addr = Code.invokeScalaObject3[Region, Row, PType, Long](
        RegistryHelpers.getClass, "stupidUnwrapStruct", r.region, coerce[Row](value), cb.emb.ecb.getPType(pt))
      new SBaseStructPointerCode(SBaseStructPointer(pt.asInstanceOf[PBaseStruct]), addr)
  }
}

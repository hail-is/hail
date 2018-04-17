package is.hail.expr.ir.functions

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s
import is.hail.asm4s._
import is.hail.expr.ir
import is.hail.expr.ir.{EmitMethodBuilder, EmitTriplet, TypeToIRIntermediateClassTag}
import is.hail.expr.types._
import is.hail.utils._

import scala.reflect.{ClassTag, classTag}

object StringFunctions extends RegistryFunctions {

  private[this] def wrapToString(mb: EmitMethodBuilder, v: Code[Long]): Code[String] = {
    Code.invokeScalaObject[Region, Long, String](TString.getClass, "loadString", mb.getArg[Region](1), v)
  }

  private[this] def convertFromString(mb: EmitMethodBuilder, v: Code[String]): Code[Long] = {
    mb.getArg[Region](1).load().appendString(v)
  }

  private[this] def registerWrappedStringScalaFunction(mname: String, argTypes: Array[Type], rType: Type)(cls: Class[_], method: String) {
    def ct(typ: Type): ClassTag[_] = typ match {
      case _: TString => classTag[String]
      case t => TypeToIRIntermediateClassTag(t)
    }

    def convertToString(mb: EmitMethodBuilder, typ: Type, v: Code[_]): Code[_] = (typ, v) match {
        case (_: TString, v2: Code[Long]) => wrapToString(mb, v2)
        case (_, v2) => v2
    }

    def convertBack(mb: EmitMethodBuilder, v: Code[_]): Code[_] = rType match {
      case _: TString => convertFromString(mb, asm4s.coerce[String](v))
      case t => v
    }

    registerCode(mname, argTypes, rType) { (mb, args) =>
      val cts = argTypes.map(ct(_).runtimeClass)
      val out = Code.invokeScalaObject(cls, method, cts, argTypes.zip(args).map { case (t, a) => convertToString(mb, t, a) } )(ct(rType))
      convertBack(mb, out)
    }
  }

  def registerWrappedStringScalaFunction(mname: String, a1: Type, rType: Type)(cls: Class[_], method: String): Unit =
    registerWrappedStringScalaFunction(mname, Array(a1), rType)(cls, method)

  def registerWrappedStringScalaFunction(mname: String, a1: Type, a2: Type, rType: Type)(cls: Class[_], method: String): Unit =
    registerWrappedStringScalaFunction(mname, Array(a1, a2), rType)(cls, method)

  def upper(s: String): String = s.toUpperCase

  def lower(s: String): String = s.toLowerCase

  def strip(s: String): String = s.trim()

  def contains(s: String, t: String): Boolean = s.contains(t)

  def startswith(s: String, t: String): Boolean = s.startsWith(t)

  def endswith(s: String, t: String): Boolean = s.endsWith(t)

  def firstMatchIn(s: String, regex: String): IndexedSeq[String] = {
    regex.r.findFirstMatchIn(s).map(_.subgroups.toArray.toFastIndexedSeq).orNull
  }

  def registerAll(): Unit = {
    val thisClass = getClass
    registerWrappedStringScalaFunction("upper", TString(), TString())(thisClass, "upper")
    registerWrappedStringScalaFunction("lower", TString(), TString())(thisClass, "lower")
    registerWrappedStringScalaFunction("strip", TString(), TString())(thisClass, "strip")
    registerWrappedStringScalaFunction("contains", TString(), TString(), TBoolean())(thisClass, "contains")
    registerWrappedStringScalaFunction("startswith", TString(), TString(), TBoolean())(thisClass, "startswith")
    registerWrappedStringScalaFunction("endswith", TString(), TString(), TBoolean())(thisClass, "endswith")

    registerCodeWithMissingness("firstMatchIn", TString(), TString(), TArray(TString())) { (mb: EmitMethodBuilder, s: EmitTriplet, r: EmitTriplet) =>
      val out: LocalRef[IndexedSeq[String]] = mb.newLocal[IndexedSeq[String]]
      val nout = new CodeNullable[IndexedSeq[String]](out)

      val srvb: StagedRegionValueBuilder = new StagedRegionValueBuilder(mb, TArray(TString()))
      val len: LocalRef[Int] = mb.newLocal[Int]
      val elt: LocalRef[String] = mb.newLocal[String]
      val nelt = new CodeNullable[String](elt)

      val setup = Code(s.setup, r.setup)
      val missing = s.m || r.m || Code(
        out := Code.invokeScalaObject[String, String, IndexedSeq[String]](thisClass,
          "firstMatchIn", wrapToString(mb, s.value[Long]), wrapToString(mb, r.value[Long])),
        nout.isNull)
      val value =
        nout.ifNull(
          ir.defaultValue(TArray(TString())),
          Code(
            len := out.invoke[Int]("size"),
            srvb.start(len),
            Code.whileLoop(srvb.arrayIdx < len,
              elt := out.invoke[Int, String]("apply", srvb.arrayIdx),
              nelt.ifNull(
                srvb.setMissing(),
                srvb.addString(elt)),
              srvb.advance()),
            srvb.end()))

      EmitTriplet(setup, missing, value)
    }

    registerCode("hamming", TString(), TString(), TInt32()) { case (mb: EmitMethodBuilder, v1: Code[Long], v2: Code[Long]) =>
      val len = mb.newLocal[Int]
      val i = mb.newLocal[Int]
      val n = mb.newLocal[Int]
      val region: Code[Region] = mb.getArg[Region](1)

      Code(
        len := TBinary.loadLength(region, v1),
        len.cne(TBinary.loadLength(region, v2)).mux(
          Code._fatal("function 'hamming' requires strings to have equal length."),
          Code(
            n := 0,
            i := 0,
            Code.whileLoop(i < len,
              region.loadByte(TBinary.bytesOffset(v1) + i.toL)
                .cne(region.loadByte(TBinary.bytesOffset(v2) + i.toL)).mux(
                n += 1,
                Code._empty[Unit]),
              i += 1),
            n)))
    }
  }
}

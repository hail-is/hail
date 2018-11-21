package is.hail.expr.ir.functions

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s
import is.hail.asm4s._
import is.hail.expr.ir
import is.hail.expr.ir.{ApplyBinaryPrimOp, EmitMethodBuilder, EmitTriplet, I32, StringLength}
import is.hail.expr.types._
import is.hail.expr.types.physical.{PArray, PBinary, PString}
import is.hail.expr.types.virtual._
import is.hail.utils._
import org.json4s.JValue
import org.json4s.jackson.JsonMethods

object StringFunctions extends RegistryFunctions {

  def upper(s: String): String = s.toUpperCase

  def lower(s: String): String = s.toLowerCase

  def strip(s: String): String = s.trim()

  def contains(s: String, t: String): Boolean = s.contains(t)

  def startswith(s: String, t: String): Boolean = s.startsWith(t)

  def endswith(s: String, t: String): Boolean = s.endsWith(t)

  def firstMatchIn(s: String, regex: String): IndexedSeq[String] = {
    regex.r.findFirstMatchIn(s).map(_.subgroups.toArray.toFastIndexedSeq).orNull
  }

  def regexMatch(regex: String, s: String): Boolean = regex.r.findFirstIn(s).isDefined

  def concat(s: String, t: String): String = s + t

  def replace(str: String, pattern1: String, pattern2: String): String =
    str.replaceAll(pattern1, pattern2)

  def split(s: String, p: String): IndexedSeq[String] = s.split(p, -1)

  def splitLimited(s: String, p: String, n: Int): IndexedSeq[String] = s.split(p, n)

  def arrayMkString(a: IndexedSeq[String], sep: String): String = a.mkString(sep)

  def setMkString(s: Set[String], sep: String): String = s.mkString(sep)

  def registerAll(): Unit = {
    val thisClass = getClass

    registerIR("[]", TString(), TInt32()) { (s, idx) =>
      // rather than do a bunch of bounds checking here, check the length of the StringSlice result - way easier
      val sName = ir.genUID()
      val sResult = ir.Ref(sName, TString())
      ir.Let(
        sName,
        ir.StringSlice(
          s,
          idx,
          ir.If(
            ir.ApplyComparisonOp(ir.EQ(TInt32()), idx, I32(-1)),
            ir.StringLength(s),
            ApplyBinaryPrimOp(ir.Add(), idx, ir.I32(1)))),
        ir.If(
          ir.ApplyComparisonOp(ir.EQ(TInt32()), ir.StringLength(sResult), ir.I32(0)),
          ir.Die("string index out of bounds", TString()), // FIXME: Die needs to take a string IR for a better error message
          sResult
        )
      )
    }
    registerIR("[:]", TString())(x => x)
    registerIR("[*:]", TString(), TInt32()) { (s, start) => ir.StringSlice(s, start, StringLength(s)) }
    registerIR("[:*]", TString(), TInt32()) { (s, end) => ir.StringSlice(s, ir.I32(0), end) }
    registerIR("[*:*]", TString(), TInt32(), TInt32()) { (s, start, end) => ir.StringSlice(s, start, end) }

    registerCode("str", tv("T"), TString()) { (mb, a) =>
      val typ = tv("T").subst()
      val annotation = boxArg(mb, typ)(a)
      val str = mb.getType(typ).invoke[Any, String]("str", annotation)
      unwrapReturn(mb, TString())(str)
    }

    registerCodeWithMissingness("json", tv("T"), TString()) { (mb, a) =>
      val typ = tv("T").subst()
      val annotation = Code(a.setup, a.m).mux(Code._null, boxArg(mb, typ)(a.v))
      val json = mb.getType(typ).invoke[Any, JValue]("toJSON", annotation)
      val str = Code.invokeScalaObject[JValue, String](JsonMethods.getClass, "compact", json)
      EmitTriplet(Code._empty, false, unwrapReturn(mb, TString())(str))
    }

    registerWrappedScalaFunction("upper", TString(), TString())(thisClass, "upper")
    registerWrappedScalaFunction("lower", TString(), TString())(thisClass, "lower")
    registerWrappedScalaFunction("strip", TString(), TString())(thisClass, "strip")
    registerWrappedScalaFunction("contains", TString(), TString(), TBoolean())(thisClass, "contains")
    registerWrappedScalaFunction("startswith", TString(), TString(), TBoolean())(thisClass, "startswith")
    registerWrappedScalaFunction("endswith", TString(), TString(), TBoolean())(thisClass, "endswith")

    registerWrappedScalaFunction("~", TString(), TString(), TBoolean())(thisClass, "regexMatch")

    registerWrappedScalaFunction("+", TString(), TString(), TString())(thisClass, "concat")

    registerWrappedScalaFunction("split", TString(), TString(), TArray(TString()))(thisClass, "split")

    registerWrappedScalaFunction("split", TString(), TString(), TInt32(), TArray(TString()))(thisClass, "splitLimited")

    registerWrappedScalaFunction("replace", TString(), TString(), TString(), TString())(thisClass, "replace")

    registerWrappedScalaFunction("mkString", TSet(TString()), TString(), TString())(thisClass, "setMkString")

    registerWrappedScalaFunction("mkString", TArray(TString()), TString(), TString())(thisClass, "arrayMkString")

    registerCodeWithMissingness("firstMatchIn", TString(), TString(), TArray(TString())) { (mb: EmitMethodBuilder, s: EmitTriplet, r: EmitTriplet) =>
      val out: LocalRef[IndexedSeq[String]] = mb.newLocal[IndexedSeq[String]]
      val nout = new CodeNullable[IndexedSeq[String]](out)

      val srvb: StagedRegionValueBuilder = new StagedRegionValueBuilder(mb, PArray(PString()))
      val len: LocalRef[Int] = mb.newLocal[Int]
      val elt: LocalRef[String] = mb.newLocal[String]
      val nelt = new CodeNullable[String](elt)

      val setup = Code(s.setup, r.setup)
      val missing = s.m || r.m || Code(
        out := Code.invokeScalaObject[String, String, IndexedSeq[String]](
          thisClass, "firstMatchIn",
          asm4s.coerce[String](wrapArg(mb, TString())(s.value[Long])),
          asm4s.coerce[String](wrapArg(mb, TString())(r.value[Long]))),
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

    registerCodeWithMissingness("hamming", TString(), TString(), TInt32()) { case (mb: EmitMethodBuilder, e1: EmitTriplet, e2: EmitTriplet) =>
      val len = mb.newLocal[Int]
      val i = mb.newLocal[Int]
      val n = mb.newLocal[Int]
      val region: Code[Region] = getRegion(mb)

      val v1 = mb.newLocal[Long]
      val v2 = mb.newLocal[Long]

      val m = Code(
        v1 := e1.value[Long],
        v2 := e2.value[Long],
        len := PBinary.loadLength(region, v1),
        len.cne(PBinary.loadLength(region, v2)))
      val v =
        Code(n := 0,
          i := 0,
          Code.whileLoop(i < len,
            region.loadByte(PBinary.bytesOffset(v1) + i.toL)
              .cne(region.loadByte(PBinary.bytesOffset(v2) + i.toL)).mux(
              n += 1,
              Code._empty[Unit]),
            i += 1),
          n)

        EmitTriplet(
          Code(e1.setup, e2.setup),
          e1.m || e2.m || m,
          m.mux(ir.defaultValue(TInt32()), v))
    }
  }
}

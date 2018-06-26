package is.hail.expr.ir.functions

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s
import is.hail.asm4s._
import is.hail.expr.ir
import is.hail.expr.ir.{EmitMethodBuilder, EmitTriplet, StringLength}
import is.hail.expr.types._
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

  def split(s: String, p: String): IndexedSeq[String] = s.split(p)

  def splitLimited(s: String, p: String, n: Int): IndexedSeq[String] = s.split(p, n)

  def arrayMkString(a: IndexedSeq[String], sep: String): String = a.mkString(sep)

  def setMkString(s: Set[String], sep: String): String = s.mkString(sep)

  def index(s: String, i: Int): String = s.slice(i, i + 1)

  def registerAll(): Unit = {
    val thisClass = getClass

    registerIR("[:]", TString())(x => x)
    registerIR("[*:]", TString(), TInt32()) { (s, start) =>
      val lenName = ir.genUID()
      val len = ir.Ref(lenName, TInt32())
      ir.Let(lenName, ir.StringLength(s),
        ir.StringSlice(
          s,
          ir.If(
            ir.ApplyComparisonOp(ir.LT(TInt32()), start, ir.I32(0)),
            UtilFunctions.max(
              ir.ApplyBinaryPrimOp(ir.Add(), len, start),
              ir.I32(0)),
            UtilFunctions.min(start, len)),
          len))
    }
    registerIR("[:*]", TString(), TInt32()) { (s, end) =>
      val lenName = ir.genUID()
      val len = ir.Ref(lenName, TInt32())
      ir.Let(lenName, ir.StringLength(s),
        ir.StringSlice(
          s,
          ir.I32(0),
          ir.If(
            ir.ApplyComparisonOp(ir.LT(TInt32()), end, ir.I32(0)),
            UtilFunctions.max(
              ir.ApplyBinaryPrimOp(ir.Add(), len, end),
              ir.I32(0)),
            UtilFunctions.min(end, len))))
    }
    registerIR("[*:*]", TString(), TInt32(), TInt32()) { (s, start, end) =>
      val lenName = ir.genUID()
      val len = ir.Ref(lenName, TInt32())
      val startName = ir.genUID()
      val startRef = ir.Ref(startName, TInt32())
      ir.Let(lenName, ir.StringLength(s),
        ir.Let(
          startName,
          ir.If(
            ir.ApplyComparisonOp(ir.LT(TInt32()), start, ir.I32(0)),
            UtilFunctions.max(
              ir.ApplyBinaryPrimOp(ir.Add(), len, start),
              ir.I32(0)),
            UtilFunctions.min(start, len)),
          ir.StringSlice(
            s,
            startRef,
            ir.If(
              ir.ApplyComparisonOp(ir.LT(TInt32()), end, ir.I32(0)),
              UtilFunctions.max(
                ir.ApplyBinaryPrimOp(ir.Add(), len, end),
                startRef),
              UtilFunctions.max(
                UtilFunctions.min(end, len),
                startRef)))))
    }

    registerIR("len", TString()) { (s) =>
      ir.StringLength(s)
    }

    registerCodeWithMissingness("str", tv("T"), TString()) { (mb, a) =>
      val typ = tv("T").subst()
      val annotation = Code(a.setup, a.m).mux(Code._null, boxArg(mb, typ)(a.v))
      val str = mb.getType(typ).invoke[Any, String]("str", annotation)
      EmitTriplet(Code._empty, false, unwrapReturn(mb, TString())(str))
    }

    registerCodeWithMissingness("json", tv("T"), TString()) { (mb, a) =>
      val typ = tv("T").subst()
      val annotation = Code(a.setup, a.m).mux(Code._null, boxArg(mb, typ)(a.v))
      val json = mb.getType(typ).invoke[Any, JValue]("toJSON", annotation)
      val str = Code.invokeScalaObject[JValue, String](JsonMethods.getClass, "compact", json)
      EmitTriplet(Code._empty, false, unwrapReturn(mb, TString())(str))
    }

    registerWrappedScalaFunction("[]", TString(), TInt32(), TString())(thisClass, "index")

    registerWrappedScalaFunction("upper", TString(), TString())(thisClass, "upper")
    registerWrappedScalaFunction("lower", TString(), TString())(thisClass, "lower")
    registerWrappedScalaFunction("strip", TString(), TString())(thisClass, "strip")
    registerWrappedScalaFunction("contains", TString(), TString(), TBoolean())(thisClass, "contains")
    registerWrappedScalaFunction("startswith", TString(), TString(), TBoolean())(thisClass, "startswith")
    registerWrappedScalaFunction("endswith", TString(), TString(), TBoolean())(thisClass, "endswith")

    registerWrappedScalaFunction("~", TString(), TString(), TBoolean())(thisClass, "regexMatch")

    registerWrappedScalaFunction("+", TString(), TString(), TString())(thisClass, "concat")

    registerIR("length", TString())(StringLength)

    registerIR("size", TString())(StringLength)

    registerWrappedScalaFunction("split", TString(), TString(), TArray(TString()))(thisClass, "split")

    registerWrappedScalaFunction("split", TString(), TString(), TInt32(), TArray(TString()))(thisClass, "splitLimited")

    registerWrappedScalaFunction("replace", TString(), TString(), TString(), TString())(thisClass, "replace")

    registerWrappedScalaFunction("mkString", TSet(TString()), TString(), TString())(thisClass, "setMkString")

    registerWrappedScalaFunction("mkString", TArray(TString()), TString(), TString())(thisClass, "arrayMkString")

    registerCodeWithMissingness("firstMatchIn", TString(), TString(), TArray(TString())) { (mb: EmitMethodBuilder, s: EmitTriplet, r: EmitTriplet) =>
      val out: LocalRef[IndexedSeq[String]] = mb.newLocal[IndexedSeq[String]]
      val nout = new CodeNullable[IndexedSeq[String]](out)

      val srvb: StagedRegionValueBuilder = new StagedRegionValueBuilder(mb, TArray(TString()))
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
        len := TBinary.loadLength(region, v1),
        len.cne(TBinary.loadLength(region, v2)))
      val v =
        Code(n := 0,
          i := 0,
          Code.whileLoop(i < len,
            region.loadByte(TBinary.bytesOffset(v1) + i.toL)
              .cne(region.loadByte(TBinary.bytesOffset(v2) + i.toL)).mux(
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

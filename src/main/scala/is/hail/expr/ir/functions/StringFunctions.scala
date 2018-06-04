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

  private[this] def registerWrappedStringScalaFunction(mname: String, argTypes: Array[Type], rType: Type)(cls: Class[_], method: String) {
    def ct(typ: Type): ClassTag[_] = typ match {
      case _: TString => classTag[String]
      case t => TypeToIRIntermediateClassTag(t)
    }

    registerCode(mname, argTypes, rType, isDet = true) { (mb, args) =>
      val cts = argTypes.map(ct(_).runtimeClass)
      val out = Code.invokeScalaObject(cls, method, cts, argTypes.zip(args).map { case (t, a) => wrapArg(mb, t)(a) }.asInstanceOf[Array[Code[_]]] )(ct(rType))
      unwrapReturn(mb, rType)(out)
    }
  }

  def registerWrappedStringScalaFunction(mname: String, a1: Type, rType: Type)(cls: Class[_], method: String): Unit =
    registerWrappedStringScalaFunction(mname, Array(a1), rType)(cls, method)

  def registerWrappedStringScalaFunction(mname: String, a1: Type, a2: Type, rType: Type)(cls: Class[_], method: String): Unit =
    registerWrappedStringScalaFunction(mname, Array(a1, a2), rType)(cls, method)

  def str(x: Int): String = x.toString

  def str(x: Long): String = x.toString

  def str(x: Float): String = x.formatted("%.5e")

  def str(x: Double): String = x.formatted("%.5e")

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

    registerWrappedStringScalaFunction("str", TInt32(), TString())(thisClass, "str")
    registerWrappedStringScalaFunction("str", TInt64(), TString())(thisClass, "str")
    registerWrappedStringScalaFunction("str", TFloat32(), TString())(thisClass, "str")
    registerWrappedStringScalaFunction("str", TFloat64(), TString())(thisClass, "str")

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
      val region: Code[Region] = mb.getArg[Region](1)

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

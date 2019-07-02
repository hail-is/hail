package is.hail.expr.ir.functions

import is.hail.asm4s
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.types.physical.{PString, PTuple}
import is.hail.utils._
import is.hail.expr.types.virtual._
import org.apache.spark.sql.Row

object UtilFunctions extends RegistryFunctions {

  def parseBoolean(s: String): Boolean = s.toBoolean

  def parseInt32(s: String): Int = s.toInt

  def parseInt64(s: String): Long = s.toLong

  def parseFloat32(s: String): Float = s match {
    case "nan" => Float.NaN
    case "inf" => Float.PositiveInfinity
    case "-inf" => Float.NegativeInfinity
    case _ => s.toFloat
  }

  def parseFloat64(s: String): Double = s match {
    case "nan" => Double.NaN
    case "inf" => Double.PositiveInfinity
    case "-inf" => Double.NegativeInfinity
    case _ => s.toDouble
  }

  def min(a: IR, b: IR): IR = If(ApplyComparisonOp(LT(a.typ), a, b), a, b)
  def max(a: IR, b: IR): IR = If(ApplyComparisonOp(GT(a.typ), a, b), a, b)

  def format(f: String, args: Row): String =
    String.format(f, args.toSeq.map(_.asInstanceOf[java.lang.Object]): _*)

  def registerAll() {
    val thisClass = getClass

    registerCode("valuesSimilar", tv("T"), tv("U"), TFloat64(), TBoolean(), TBoolean()) {
      case (er, (lT, l), (rT, r), (tolT, tolerance), (absT, absolute)) =>
        assert(lT.virtualType == rT.virtualType)
        val lb = boxArg(er, lT)(l)
        val rb = boxArg(er, rT)(r)
        er.mb.getType(lT.virtualType).invoke[Any, Any, Double, Boolean, Boolean]("valuesSimilar", lb, rb, tolerance, absolute)
    }

    registerCode[Int]("triangle", TInt32(), TInt32()) { case (_, (nT, n: Code[Int])) => (n * (n + 1)) / 2 }

    registerCode[Boolean]("toInt32", TBoolean(), TInt32()) { case (_, (xT, x: Code[Boolean])) => x.toI }
    registerCode[Boolean]("toInt64", TBoolean(), TInt64()) { case (_, (xT, x: Code[Boolean])) => x.toI.toL }
    registerCode[Boolean]("toFloat32", TBoolean(), TFloat32()) { case (_, (xT, x: Code[Boolean])) => x.toI.toF }
    registerCode[Boolean]("toFloat64", TBoolean(), TFloat64()) { case (_, (xT, x: Code[Boolean])) => x.toI.toD }
    registerCode("toInt32", TString(), TInt32()) {
      case (r, (xT: PString, x: Code[Long])) =>
        val s = asm4s.coerce[String](wrapArg(r, xT)(x))
        Code.invokeScalaObject[String, Int](thisClass, "parseInt32", s)
    }
    registerCode("toInt64", TString(), TInt64()) {
      case (r, (xT: PString, x: Code[Long])) =>
        val s = asm4s.coerce[String](wrapArg(r, xT)(x))
        Code.invokeScalaObject[String, Long](thisClass, "parseInt64", s)
    }
    registerCode("toFloat32", TString(), TFloat32()) {
      case (r, (xT: PString, x: Code[Long])) =>
        val s = asm4s.coerce[String](wrapArg(r, xT)(x))
        Code.invokeScalaObject[String, Float](thisClass, "parseFloat32", s)
    }
    registerCode("toFloat64", TString(), TFloat64()) {
      case (r, (xT: PString, x: Code[Long])) =>
        val s = asm4s.coerce[String](wrapArg(r, xT)(x))
        Code.invokeScalaObject[String, Double](thisClass, "parseFloat64", s)
    }
    registerCode("toBoolean", TString(), TBoolean()) {
      case (r, (xT: PString, x: Code[Long])) =>
        val s = asm4s.coerce[String](wrapArg(r, xT)(x))
        Code.invokeScalaObject[String, Boolean](thisClass, "parseBoolean", s)
    }

    registerIR("min", tv("T"), tv("T"), tv("T"))(min)
    registerIR("max", tv("T"), tv("T"), tv("T"))(max)

    registerCode("format", TString(), tv("T", "tuple"), TString()) {
      case (r, (fmtT: PString, format: Code[Long]), (argsT: PTuple, args: Code[Long])) =>
        r.region.appendString(Code.invokeScalaObject[String, Row, String](thisClass, "format",
          asm4s.coerce[String](wrapArg(r, fmtT)(format)),
          Code.checkcast[Row](asm4s.coerce[java.lang.Object](wrapArg(r, argsT)(args)))))
    }

    registerCodeWithMissingness("&&", TBoolean(), TBoolean(), TBoolean()) {
      case (er, (lT, l), (rT, r)) =>
        val lm = Code(l.setup, l.m)
        val rm = Code(r.setup, r.m)

        val lv = l.value[Boolean]
        val rv = r.value[Boolean]

        val m = er.mb.newLocal[Boolean]
        val v = er.mb.newLocal[Boolean]
        val setup = Code(m := lm, v := !m && lv)
        val missing = m.mux(rm || rv, v && (rm || Code(v := rv, false)))
        val value = v

        EmitTriplet(setup, missing, value)
    }

    registerCodeWithMissingness("||", TBoolean(), TBoolean(), TBoolean()) {
      case (er, (lT, l), (rT, r)) =>
        val lm = Code(l.setup, l.m)
        val rm = Code(r.setup, r.m)

        val lv = l.value[Boolean]
        val rv = r.value[Boolean]

        val m = er.mb.newLocal[Boolean]
        val v = er.mb.newLocal[Boolean]
        val setup = Code(m := lm, v := m || lv)
        val missing = m.mux(rm || !rv, !v && (rm || Code(v := rv, false)))
        val value = v

        EmitTriplet(setup, missing, value)
    }
  }
}
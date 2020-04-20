package is.hail.expr.ir.functions

import is.hail.asm4s
import is.hail.asm4s.{coerce => _, _}
import is.hail.expr.ir._
import is.hail.expr.types.physical._
import is.hail.utils._
import is.hail.expr.types.virtual._
import org.apache.spark.sql.Row

import scala.reflect.ClassTag

object UtilFunctions extends RegistryFunctions {

  def parseBoolean(s: String): Boolean = s.toBoolean

  def parseInt32(s: String): Int = s.toInt

  def parseInt64(s: String): Long = s.toLong

  private val NAN = 1
  private val POS_INF = 2
  private val NEG_INF = 3

  def parseSpecialNum(s: String): Int = s.length match {
    case 3 if s equalsCI "nan" => NAN
    case 4 if (s equalsCI "+nan") || (s equalsCI "-nan") => NAN
    case 3 if s equalsCI "inf" => POS_INF
    case 4 if s equalsCI "+inf" => POS_INF
    case 4 if s equalsCI "-inf" => NEG_INF
    case 8 if s equalsCI "infinity" => POS_INF
    case 9 if s equalsCI "+infinity" => POS_INF
    case 9 if s equalsCI "-infinity" => NEG_INF
    case _ => 0
  }

  def parseFloat32(s: String): Float = parseSpecialNum(s) match {
    case NAN => Float.NaN
    case POS_INF => Float.PositiveInfinity
    case NEG_INF => Float.NegativeInfinity
    case _ => s.toFloat
  }

  def parseFloat64(s: String): Double = parseSpecialNum(s) match {
    case NAN => Double.NaN
    case POS_INF => Double.PositiveInfinity
    case NEG_INF => Double.NegativeInfinity
    case _ => s.toDouble
  }

  def isValidBoolean(s: String): Boolean =
    (s equalsCI "true") || (s equalsCI "false")

  def isValidInt32(s: String): Boolean =
    try { s.toInt; true } catch { case _: NumberFormatException => false }

  def isValidInt64(s: String): Boolean =
    try { s.toLong; true } catch { case _: NumberFormatException => false }

  def isValidFloat32(s: String): Boolean = parseSpecialNum(s) match {
    case 0 => try { s.toFloat; true } catch { case _: NumberFormatException => false }
    case _ => true
  }

  def isValidFloat64(s: String): Boolean = parseSpecialNum(s) match {
    case 0 => try { s.toDouble; true } catch { case _: NumberFormatException => false }
    case _ => true
  }

  def min_ignore_missing(l: Int, lMissing: Boolean, r: Int, rMissing: Boolean): Int =
    if (lMissing) r else if (rMissing) l else Math.min(l, r)

  def min_ignore_missing(l: Long, lMissing: Boolean, r: Long, rMissing: Boolean): Long =
    if (lMissing) r else if (rMissing) l else Math.min(l, r)

  def min_ignore_missing(l: Float, lMissing: Boolean, r: Float, rMissing: Boolean): Float =
    if (lMissing) r else if (rMissing) l else Math.min(l, r)

  def min_ignore_missing(l: Double, lMissing: Boolean, r: Double, rMissing: Boolean): Double =
    if (lMissing) r else if (rMissing) l else Math.min(l, r)

  def max_ignore_missing(l: Int, lMissing: Boolean, r: Int, rMissing: Boolean): Int =
    if (lMissing) r else if (rMissing) l else Math.max(l, r)

  def max_ignore_missing(l: Long, lMissing: Boolean, r: Long, rMissing: Boolean): Long =
    if (lMissing) r else if (rMissing) l else Math.max(l, r)

  def max_ignore_missing(l: Float, lMissing: Boolean, r: Float, rMissing: Boolean): Float =
    if (lMissing) r else if (rMissing) l else Math.max(l, r)

  def max_ignore_missing(l: Double, lMissing: Boolean, r: Double, rMissing: Boolean): Double =
    if (lMissing) r else if (rMissing) l else Math.max(l, r)

  def nanmax(l: Double, r: Double): Double =
    if (java.lang.Double.isNaN(l)) r else if (java.lang.Double.isNaN(r)) l else Math.max(l, r)

  def nanmax(l: Float, r: Float): Float =
    if (java.lang.Float.isNaN(l)) r else if (java.lang.Float.isNaN(r)) l else Math.max(l, r)

  def nanmin(l: Double, r: Double): Double =
    if (java.lang.Double.isNaN(l)) r else if (java.lang.Double.isNaN(r)) l else Math.min(l, r)

  def nanmin(l: Float, r: Float): Float =
    if (java.lang.Float.isNaN(l)) r else if (java.lang.Float.isNaN(r)) l else Math.min(l, r)

  def nanmin_ignore_missing(l: Float, lMissing: Boolean, r: Float, rMissing: Boolean): Float =
    if (lMissing) r else if (rMissing) l else nanmin(l, r)

  def nanmin_ignore_missing(l: Double, lMissing: Boolean, r: Double, rMissing: Boolean): Double =
    if (lMissing) r else if (rMissing) l else nanmin(l, r)

  def nanmax_ignore_missing(l: Float, lMissing: Boolean, r: Float, rMissing: Boolean): Float =
    if (lMissing) r else if (rMissing) l else nanmax(l, r)

  def nanmax_ignore_missing(l: Double, lMissing: Boolean, r: Double, rMissing: Boolean): Double =
    if (lMissing) r else if (rMissing) l else nanmax(l, r)

  def intMin(a: IR, b: IR): IR = If(ApplyComparisonOp(LT(a.typ), a, b), a, b)

  def intMax(a: IR, b: IR): IR = If(ApplyComparisonOp(GT(a.typ), a, b), a, b)

  def format(f: String, args: Row): String =
    String.format(f, args.toSeq.map(_.asInstanceOf[java.lang.Object]): _*)

  def registerAll() {
    val thisClass = getClass

    registerCode4("valuesSimilar", tv("T"), tv("U"), TFloat64, TBoolean, TBoolean, {
      case(_: Type, _: PType, _: PType, _: PType, _: PType) => PBoolean()
    }) {
      case (er, rt, (lT, l), (rT, r), (tolT, tolerance), (absT, absolute)) =>
        assert(lT.virtualType == rT.virtualType)
        val lb = boxArg(er, lT)(l)
        val rb = boxArg(er, rT)(r)
        er.mb.getType(lT.virtualType).invoke[Any, Any, Double, Boolean, Boolean]("valuesSimilar", lb, rb, tolerance, absolute)
    }

    registerCode1[Int]("triangle", TInt32, TInt32, (_: Type, n: PType) => n) { case (_, rt, (nT, n: Code[Int])) =>
      Code.memoize(n, "triangle_n") { n =>
        (n * (n + 1)) / 2
      }
    }

    registerCode1[Boolean]("toInt32", TBoolean, TInt32, (_: Type, _: PType) => PInt32()) { case (_, rt, (xT, x: Code[Boolean])) => x.toI }
    registerCode1[Boolean]("toInt64", TBoolean, TInt64, (_: Type, _: PType) => PInt64()) { case (_, rt, (xT, x: Code[Boolean])) => x.toI.toL }
    registerCode1[Boolean]("toFloat32", TBoolean, TFloat32, (_: Type, _: PType) => PFloat32()) { case (_, rt, (xT, x: Code[Boolean])) => x.toI.toF }
    registerCode1[Boolean]("toFloat64", TBoolean, TFloat64, (_: Type, _: PType) => PFloat64()) { case (_, rt, (xT, x: Code[Boolean])) => x.toI.toD }

    for ((name, t, rpt, ct) <- Seq[(String, Type, PType, ClassTag[_])](
      ("Boolean", TBoolean, PBoolean(), implicitly[ClassTag[Boolean]]),
      ("Int32", TInt32, PInt32(), implicitly[ClassTag[Int]]),
      ("Int64", TInt64, PInt64(), implicitly[ClassTag[Long]]),
      ("Float64", TFloat64, PFloat64(), implicitly[ClassTag[Double]]),
      ("Float32", TFloat32, PFloat32(), implicitly[ClassTag[Float]])
    )) {
      val ctString: ClassTag[String] = implicitly
      registerCode1(s"to$name", TString, t, (_: Type, _: PType) => rpt) {
        case (r, rt, (xT: PString, x: Code[Long])) =>
          val s = asm4s.coerce[String](wrapArg(r, xT)(x))
          Code.invokeScalaObject1(thisClass, s"parse$name", s)(ctString, ct)
      }
      registerEmitCode1(s"to${name}OrMissing", TString, t, (_: Type, xPT: PType) => rpt.setRequired(xPT.required)) {
        case (r, rt, x) =>
          val s = r.mb.newLocal[String]()
          val m = r.mb.newLocal[Boolean]()
          EmitCode(
            Code(x.setup, m := x.m, s := m.mux(Code._null[String], asm4s.coerce[String](wrapArg(r, x.pt)(x.v)))),
            (m || !Code.invokeScalaObject1[String, Boolean](thisClass, s"isValid$name", s)),
            PCode(rt, Code.invokeScalaObject1(thisClass, s"parse$name", s)(ctString, ct)))
      }
    }

    Array(TInt32, TInt64).foreach { t =>
      registerIR2("min", t, t, t)((_, a, b) => intMin(a, b))
      registerIR2("max", t, t, t)((_, a, b) => intMax(a, b))
    }

    Array("min", "max").foreach { name =>
      registerCode2(name, TFloat32, TFloat32, TFloat32, (_: Type, _: PType, _: PType) => PFloat32()) {
        case (r, rt, (t1, v1: Code[Float]), (t2, v2: Code[Float])) =>
          Code.invokeStatic2[Math, Float, Float, Float](name, v1, v2)
      }

      registerCode2(name, TFloat64, TFloat64, TFloat64, (_: Type, _: PType, _: PType) => PFloat64()) {
        case (r, rt, (t1, v1: Code[Double]), (t2, v2: Code[Double])) =>
          Code.invokeStatic2[Math, Double, Double, Double](name, v1, v2)
      }

      val ignoreMissingName = name + "_ignore_missing"
      val ignoreNanName = "nan" + name
      val ignoreBothName = ignoreNanName + "_ignore_missing"

      registerCode2(ignoreNanName, TFloat32, TFloat32, TFloat32, (_: Type, _: PType, _: PType) => PFloat32()) {
        case (r, rt, (t1, v1: Code[Float]), (t2, v2: Code[Float])) =>
          Code.invokeScalaObject2[Float, Float, Float](thisClass, ignoreNanName, v1, v2)
      }

      registerCode2(ignoreNanName, TFloat64, TFloat64, TFloat64, (_: Type, _: PType, _: PType) => PFloat64()) {
        case (r, rt, (t1, v1: Code[Double]), (t2, v2: Code[Double])) =>
          Code.invokeScalaObject2[Double, Double, Double](thisClass, ignoreNanName, v1, v2)
      }

      def ignoreMissingTriplet[T](rt: PType, v1: EmitCode, v2: EmitCode, name: String)(implicit ct: ClassTag[T], ti: TypeInfo[T]): EmitCode = {
        val m1 = Code.newLocal[Boolean]("min_max_igno_miss_m1")
        val m2 = Code.newLocal[Boolean]("min_max_igno_miss_m2")
        EmitCode(
          Code(v1.setup, v2.setup, m1 := v1.m, m2 := v2.m),
          m1 && m2,
          PCode(rt, Code.invokeScalaObject4[T, Boolean, T, Boolean, T](thisClass, name,
            m1.mux(coerce[T](defaultValue(ti)), v1.value[T]), m1,
            m2.mux(coerce[T](defaultValue(ti)), v2.value[T]), m2)))
      }

      registerEmitCode2(ignoreMissingName, TInt32, TInt32, TInt32, (_: Type, t1: PType, t2: PType) => PInt32(t1.required && t2.required)) {
        case (r, rt, v1, v2) => ignoreMissingTriplet[Int](rt, v1, v2, ignoreMissingName)
      }

      registerEmitCode2(ignoreMissingName, TInt64, TInt64, TInt64, (_: Type, t1: PType, t2: PType) => PInt64(t1.required && t2.required)) {
        case (r, rt, v1, v2) => ignoreMissingTriplet[Long](rt, v1, v2, ignoreMissingName)
      }

      registerEmitCode2(ignoreMissingName, TFloat32, TFloat32, TFloat32, (_: Type, t1: PType, t2: PType) => PFloat32(t1.required && t2.required)) {
        case (r, rt, v1, v2) => ignoreMissingTriplet[Float](rt, v1, v2, ignoreMissingName)
      }

      registerEmitCode2(ignoreMissingName, TFloat64, TFloat64, TFloat64, (_: Type, t1: PType, t2: PType) => PFloat64(t1.required && t2.required)) {
        case (r, rt, v1, v2) => ignoreMissingTriplet[Double](rt, v1, v2, ignoreMissingName)
      }

      registerEmitCode2(ignoreBothName, TFloat32, TFloat32, TFloat32, (_: Type, t1: PType, t2: PType) => PFloat32(t1.required && t2.required)) {
        case (r, rt, v1, v2) => ignoreMissingTriplet[Float](rt, v1, v2, ignoreBothName)
      }

      registerEmitCode2(ignoreBothName, TFloat64, TFloat64, TFloat64, (_: Type, t1: PType, t2: PType) => PFloat64(t1.required && t2.required)) {
        case (r, rt, v1, v2) => ignoreMissingTriplet[Double](rt, v1, v2, ignoreBothName)
      }
    }

    registerCode2("format", TString, tv("T", "tuple"), TString, (_: Type, _: PType, _: PType) => PCanonicalString()) {
      case (r, rt, (fmtT: PString, format: Code[Long]), (argsT: PTuple, args: Code[Long])) =>
        unwrapReturn(r, rt)(Code.invokeScalaObject2[String, Row, String](thisClass, "format",
          asm4s.coerce[String](wrapArg(r, fmtT)(format)),
          Code.checkcast[Row](asm4s.coerce[java.lang.Object](wrapArg(r, argsT)(args)))))
    }

    registerEmitCode2("land", TBoolean, TBoolean, TBoolean, (_: Type, _: PType, _: PType) => PBoolean()) {
      case (er, rt, l, r) =>
        val lv = l.value[Boolean]
        val rv = r.value[Boolean]

        // 00 ... 00 rv rm lv lm
        val w = er.mb.newLocal[Int]()

        // m/m, t/m, m/t
        val M = const((1 << 5) | (1 << 6) | (1 << 9))

        val setup = Code(l.setup,
          w := l.m.mux(const(1),
            lv.mux(
              const(2),
              const(0))),
          w.cne(0).mux(
            Code(
              r.setup,
              w := w | r.m.mux(const(4),
              rv.mux(
                const(8),
                const(0)))),
            Code._empty))

        EmitCode(setup,
          ((M >> w) & 1).cne(0),
          PCode(rt, w.ceq(10)))
    }

    registerEmitCode2("lor", TBoolean, TBoolean, TBoolean, (_: Type, _: PType, _: PType) => PBoolean()) {
      case (er, rt, l, r) =>
        val lv = l.value[Boolean]
        val rv = r.value[Boolean]

        // 00 ... 00 rv rm lv lm
        val w = er.mb.newLocal[Int]()

        // m/m, f/m, m/f
        val M = const((1 << 5) | (1 << 1) | (1 << 4))

        val setup = Code(l.setup,
          w := l.m.mux(const(1),
            lv.mux(
              const(2),
              const(0))),
          w.cne(2).mux(
            Code(
              r.setup,
              w := w | r.m.mux(const(4),
                rv.mux(
                  const(8),
                  const(0)))),
            Code._empty))

        EmitCode(setup,
          ((M >> w) & 1).cne(0),
          PCode(rt, w.cne(0)))
    }
  }
}

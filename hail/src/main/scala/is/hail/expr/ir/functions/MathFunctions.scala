package is.hail.expr.ir.functions

import is.hail.asm4s.Code
import is.hail.expr.ir._
import is.hail.stats._
import is.hail.types.physical.stypes.concrete._
import is.hail.types.physical.stypes.interfaces.primitive
import is.hail.types.physical.stypes.primitives._
import is.hail.types.virtual._
import is.hail.utils._

import org.apache.commons.math3.special.Gamma

object MathFunctions extends RegistryFunctions {
  def log(x: Double, b: Double): Double = math.log(x) / math.log(b)

  // This does a truncating log2, always rounnds down
  def log2(x: Int): Int = {
    var v = x
    var r = if (v > 0xffff) 16 else 0
    v >>= r
    if (v > 0xff) { v >>= 8; r |= 8 }
    if (v > 0xf) { v >>= 4; r |= 4 }
    if (v > 0x3) { v >>= 2; r |= 2 }
    r |= v >> 1
    r
  }

  def roundToNextPowerOf2(x: Int): Int = {
    var v = x - 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    v + 1
  }

  def gamma(x: Double): Double = Gamma.gamma(x)

  def floor(x: Float): Float = math.floor(x).toFloat

  def floor(x: Double): Double = math.floor(x)

  def ceil(x: Float): Float = math.ceil(x).toFloat

  def ceil(x: Double): Double = math.ceil(x)

  def mod(x: Int, y: Int): Int = {
    if (y == 0)
      fatal(s"$x % 0: modulo by zero", ErrorIDs.NO_ERROR)
    java.lang.Math.floorMod(x, y)
  }

  def mod(x: Long, y: Long): Long = {
    if (y == 0L)
      fatal(s"$x % 0: modulo by zero", ErrorIDs.NO_ERROR)
    java.lang.Math.floorMod(x, y)
  }

  def mod(x: Float, y: Float): Float = {
    if (y == 0.0)
      fatal(s"$x % 0: modulo by zero", ErrorIDs.NO_ERROR)
    val t = x % y
    if (t < 0) t + y else t
  }

  def mod(x: Double, y: Double): Double = {
    if (y == 0.0)
      fatal(s"$x % 0: modulo by zero", ErrorIDs.NO_ERROR)
    val t = x % y
    if (t < 0) t + y else t
  }

  def pow(x: Int, y: Int): Double = math.pow(x, y)

  def pow(x: Long, y: Long): Double = math.pow(x, y)

  def pow(x: Float, y: Float): Double = math.pow(x, y)

  def pow(x: Double, y: Double): Double = math.pow(x, y)

  def floorDiv(x: Int, y: Int): Int = {
    if (y == 0)
      fatal(s"$x // 0: integer division by zero", ErrorIDs.NO_ERROR)
    java.lang.Math.floorDiv(x, y)
  }

  def floorDiv(x: Long, y: Long): Long = {
    if (y == 0L)
      fatal(s"$x // 0: integer division by zero", ErrorIDs.NO_ERROR)
    java.lang.Math.floorDiv(x, y)
  }

  def floorDiv(x: Float, y: Float): Float = math.floor(x / y).toFloat

  def floorDiv(x: Double, y: Double): Double = math.floor(x / y)

  def approxEqual(x: Double, y: Double, tolerance: Double, absolute: Boolean, nanSame: Boolean)
    : Boolean = {
    val withinTol =
      if (absolute)
        math.abs(x - y) <= tolerance
      else
        D_==(x, y, tolerance)
    x == y || withinTol || (nanSame && x.isNaN && y.isNaN)
  }

  def irentropy(s: String): Double = entropy(s)

  val mathPackageClass: Class[_] = Class.forName("scala.math.package$")

  def registerAll() {
    val thisClass = getClass
    val statsPackageClass = Class.forName("is.hail.stats.package$")
    val jMathClass = classOf[java.lang.Math]
    val jIntegerClass = classOf[java.lang.Integer]
    val jFloatClass = classOf[java.lang.Float]
    val jDoubleClass = classOf[java.lang.Double]

    // numeric conversions
    registerIR1("toInt32", tnum("T"), TInt32)((_, x, _) => Cast((x), TInt32))
    registerIR1("toInt64", tnum("T"), TInt64)((_, x, _) => Cast(x, TInt64))
    registerIR1("toFloat32", tnum("T"), TFloat32)((_, x, _) => Cast(x, TFloat32))
    registerIR1("toFloat64", tnum("T"), TFloat64)((_, x, _) => Cast(x, TFloat64))

    registerScalaFunction("abs", Array(TInt32), TInt32, null)(mathPackageClass, "abs")
    registerScalaFunction("abs", Array(TInt64), TInt64, null)(mathPackageClass, "abs")
    registerScalaFunction("abs", Array(TFloat32), TFloat32, null)(mathPackageClass, "abs")
    registerScalaFunction("abs", Array(TFloat64), TFloat64, null)(mathPackageClass, "abs")

    registerScalaFunction("pow", Array(TInt32, TInt32), TFloat64, null)(thisClass, "pow")
    registerScalaFunction("pow", Array(TInt64, TInt64), TFloat64, null)(thisClass, "pow")
    registerScalaFunction("pow", Array(TFloat32, TFloat32), TFloat64, null)(thisClass, "pow")
    registerScalaFunction("pow", Array(TFloat64, TFloat64), TFloat64, null)(thisClass, "pow")

    registerScalaFunction("exp", Array(TFloat64), TFloat64, null)(mathPackageClass, "exp")
    registerScalaFunction("log10", Array(TFloat64), TFloat64, null)(mathPackageClass, "log10")
    registerScalaFunction("sqrt", Array(TFloat64), TFloat64, null)(mathPackageClass, "sqrt")
    registerScalaFunction("log", Array(TFloat64), TFloat64, null)(mathPackageClass, "log")
    registerScalaFunction("log", Array(TFloat64, TFloat64), TFloat64, null)(thisClass, "log")
    registerScalaFunction("log2", Array(TInt32), TInt32, null)(thisClass, "log2")
    registerScalaFunction("roundToNextPowerOf2", Array(TInt32), TInt32, null)(
      thisClass,
      "roundToNextPowerOf2",
    )
    registerScalaFunction("gamma", Array(TFloat64), TFloat64, null)(thisClass, "gamma")
    registerScalaFunction("binomTest", Array(TInt32, TInt32, TFloat64, TInt32), TFloat64, null)(
      statsPackageClass,
      "binomTest",
    )

    registerScalaFunction("dbeta", Array(TFloat64, TFloat64, TFloat64), TFloat64, null)(
      statsPackageClass,
      "dbeta",
    )

    registerScalaFunction("dnorm", Array(TFloat64), TFloat64, null)(statsPackageClass, "dnorm")
    registerScalaFunction("dnorm", Array(TFloat64, TFloat64, TFloat64, TBoolean), TFloat64, null)(
      statsPackageClass,
      "dnorm",
    )

    registerScalaFunction("pnorm", Array(TFloat64), TFloat64, null)(statsPackageClass, "pnorm")
    registerScalaFunction(
      "pnorm",
      Array(TFloat64, TFloat64, TFloat64, TBoolean, TBoolean),
      TFloat64,
      null,
    )(statsPackageClass, "pnorm")

    registerScalaFunction("qnorm", Array(TFloat64), TFloat64, null)(statsPackageClass, "qnorm")
    registerScalaFunction(
      "qnorm",
      Array(TFloat64, TFloat64, TFloat64, TBoolean, TBoolean),
      TFloat64,
      null,
    )(statsPackageClass, "qnorm")

    registerScalaFunction("pT", Array(TFloat64, TFloat64, TBoolean, TBoolean), TFloat64, null)(
      statsPackageClass,
      "pT",
    )
    registerScalaFunction(
      "pF",
      Array(TFloat64, TFloat64, TFloat64, TBoolean, TBoolean),
      TFloat64,
      null,
    )(statsPackageClass, "pF")

    registerScalaFunction("dpois", Array(TFloat64, TFloat64), TFloat64, null)(
      statsPackageClass,
      "dpois",
    )
    registerScalaFunction("dpois", Array(TFloat64, TFloat64, TBoolean), TFloat64, null)(
      statsPackageClass,
      "dpois",
    )

    registerScalaFunction("ppois", Array(TFloat64, TFloat64), TFloat64, null)(
      statsPackageClass,
      "ppois",
    )
    registerScalaFunction("ppois", Array(TFloat64, TFloat64, TBoolean, TBoolean), TFloat64, null)(
      statsPackageClass,
      "ppois",
    )

    registerScalaFunction("qpois", Array(TFloat64, TFloat64), TInt32, null)(
      statsPackageClass,
      "qpois",
    )
    registerScalaFunction("qpois", Array(TFloat64, TFloat64, TBoolean, TBoolean), TInt32, null)(
      statsPackageClass,
      "qpois",
    )

    registerScalaFunction("dchisq", Array(TFloat64, TFloat64), TFloat64, null)(
      statsPackageClass,
      "dchisq",
    )
    registerScalaFunction("dchisq", Array(TFloat64, TFloat64, TBoolean), TFloat64, null)(
      statsPackageClass,
      "dchisq",
    )

    registerScalaFunction("dnchisq", Array(TFloat64, TFloat64, TFloat64), TFloat64, null)(
      statsPackageClass,
      "dnchisq",
    )
    registerScalaFunction("dnchisq", Array(TFloat64, TFloat64, TFloat64, TBoolean), TFloat64, null)(
      statsPackageClass,
      "dnchisq",
    )

    registerScalaFunction("pchisqtail", Array(TFloat64, TFloat64), TFloat64, null)(
      statsPackageClass,
      "pchisqtail",
    )
    registerScalaFunction(
      "pchisqtail",
      Array(TFloat64, TFloat64, TBoolean, TBoolean),
      TFloat64,
      null,
    )(statsPackageClass, "pchisqtail")

    registerScalaFunction("pnchisqtail", Array(TFloat64, TFloat64, TFloat64), TFloat64, null)(
      statsPackageClass,
      "pnchisqtail",
    )
    registerScalaFunction(
      "pnchisqtail",
      Array(TFloat64, TFloat64, TFloat64, TBoolean, TBoolean),
      TFloat64,
      null,
    )(statsPackageClass, "pnchisqtail")

    registerScalaFunction("qchisqtail", Array(TFloat64, TFloat64), TFloat64, null)(
      statsPackageClass,
      "qchisqtail",
    )
    registerScalaFunction(
      "qchisqtail",
      Array(TFloat64, TFloat64, TBoolean, TBoolean),
      TFloat64,
      null,
    )(statsPackageClass, "qchisqtail")

    registerScalaFunction("qnchisqtail", Array(TFloat64, TFloat64, TFloat64), TFloat64, null)(
      statsPackageClass,
      "qnchisqtail",
    )
    registerScalaFunction(
      "qnchisqtail",
      Array(TFloat64, TFloat64, TFloat64, TBoolean, TBoolean),
      TFloat64,
      null,
    )(statsPackageClass, "qnchisqtail")

    registerSCode7(
      "pgenchisq",
      TFloat64,
      TArray(TFloat64),
      TArray(TInt32),
      TArray(TFloat64),
      TFloat64,
      TInt32,
      TFloat64,
      DaviesAlgorithm.pType.virtualType,
      (_, _, _, _, _, _, _, _) => DaviesAlgorithm.pType.sType,
    ) {
      case (
            r,
            cb,
            rt,
            x: SFloat64Value,
            _w: SIndexablePointerValue,
            _k: SIndexablePointerValue,
            _lam: SIndexablePointerValue,
            sigma: SFloat64Value,
            maxIterations: SInt32Value,
            minAccuracy: SFloat64Value,
            _,
          ) =>
        val w = _w.castToArray(cb)
        val k = _k.castToArray(cb)
        val lam = _lam.castToArray(cb)

        val res = cb.newLocal[DaviesResultForPython](
          "pgenchisq_result",
          Code.invokeScalaObject7[
            Double,
            IndexedSeq[Double],
            IndexedSeq[Int],
            IndexedSeq[Double],
            Double,
            Int,
            Double,
            DaviesResultForPython,
          ](
            statsPackageClass,
            "pgenchisq",
            x.value,
            Code.checkcast[IndexedSeq[Double]](svalueToJavaValue(cb, r.region, w)),
            Code.checkcast[IndexedSeq[Int]](svalueToJavaValue(cb, r.region, k)),
            Code.checkcast[IndexedSeq[Double]](svalueToJavaValue(cb, r.region, lam)),
            sigma.value,
            maxIterations.value,
            minAccuracy.value,
          ),
        )

        DaviesAlgorithm.pType.constructFromFields(
          cb,
          r.region,
          FastSeq(
            EmitValue.present(primitive(cb.memoize(res.invoke[Double]("value")))),
            EmitValue.present(primitive(cb.memoize(res.invoke[Int]("nIterations")))),
            EmitValue.present(primitive(cb.memoize(res.invoke[Boolean]("converged")))),
            EmitValue.present(primitive(cb.memoize(res.invoke[Int]("fault")))),
          ),
          deepCopy = false,
        )
    }

    registerScalaFunction("floor", Array(TFloat32), TFloat32, null)(thisClass, "floor")
    registerScalaFunction("floor", Array(TFloat64), TFloat64, null)(thisClass, "floor")

    registerScalaFunction("ceil", Array(TFloat32), TFloat32, null)(thisClass, "ceil")
    registerScalaFunction("ceil", Array(TFloat64), TFloat64, null)(thisClass, "ceil")

    registerScalaFunction("mod", Array(TInt32, TInt32), TInt32, null)(thisClass, "mod")
    registerScalaFunction("mod", Array(TInt64, TInt64), TInt64, null)(thisClass, "mod")
    registerScalaFunction("mod", Array(TFloat32, TFloat32), TFloat32, null)(thisClass, "mod")
    registerScalaFunction("mod", Array(TFloat64, TFloat64), TFloat64, null)(thisClass, "mod")

    registerJavaStaticFunction("isnan", Array(TFloat32), TBoolean, null)(jFloatClass, "isNaN")
    registerJavaStaticFunction("isnan", Array(TFloat64), TBoolean, null)(jDoubleClass, "isNaN")

    registerJavaStaticFunction("is_finite", Array(TFloat32), TBoolean, null)(
      jFloatClass,
      "isFinite",
    )
    registerJavaStaticFunction("is_finite", Array(TFloat64), TBoolean, null)(
      jDoubleClass,
      "isFinite",
    )

    registerJavaStaticFunction("is_infinite", Array(TFloat32), TBoolean, null)(
      jFloatClass,
      "isInfinite",
    )
    registerJavaStaticFunction("is_infinite", Array(TFloat64), TBoolean, null)(
      jDoubleClass,
      "isInfinite",
    )

    registerJavaStaticFunction("sign", Array(TInt32), TInt32, null)(jIntegerClass, "signum")
    registerScalaFunction("sign", Array(TInt64), TInt64, null)(mathPackageClass, "signum")
    registerJavaStaticFunction("sign", Array(TFloat32), TFloat32, null)(jMathClass, "signum")
    registerJavaStaticFunction("sign", Array(TFloat64), TFloat64, null)(jMathClass, "signum")

    registerScalaFunction(
      "approxEqual",
      Array(TFloat64, TFloat64, TFloat64, TBoolean, TBoolean),
      TBoolean,
      null,
    )(thisClass, "approxEqual")

    registerWrappedScalaFunction1("entropy", TString, TFloat64, null)(thisClass, "irentropy")

    registerSCode4(
      "fisher_exact_test",
      TInt32,
      TInt32,
      TInt32,
      TInt32,
      fetStruct.virtualType,
      (_, _, _, _, _) => fetStruct.sType,
    ) { case (r, cb, rt, a: SInt32Value, b: SInt32Value, c: SInt32Value, d: SInt32Value, _) =>
      val res = cb.newLocal[Array[Double]](
        "fisher_exact_test_res",
        Code.invokeScalaObject4[Int, Int, Int, Int, Array[Double]](
          statsPackageClass,
          "fisherExactTest",
          a.value,
          b.value,
          c.value,
          d.value,
        ),
      )

      fetStruct.constructFromFields(
        cb,
        r.region,
        FastSeq(
          EmitValue.present(primitive(cb.memoize(res(0)))),
          EmitValue.present(primitive(cb.memoize(res(1)))),
          EmitValue.present(primitive(cb.memoize(res(2)))),
          EmitValue.present(primitive(cb.memoize(res(3)))),
        ),
        deepCopy = false,
      )
    }

    registerSCode4(
      "chi_squared_test",
      TInt32,
      TInt32,
      TInt32,
      TInt32,
      chisqStruct.virtualType,
      (_, _, _, _, _) => chisqStruct.sType,
    ) { case (r, cb, rt, a: SInt32Value, b: SInt32Value, c: SInt32Value, d: SInt32Value, _) =>
      val res = cb.newLocal[Array[Double]](
        "chi_squared_test_res",
        Code.invokeScalaObject4[Int, Int, Int, Int, Array[Double]](
          statsPackageClass,
          "chiSquaredTest",
          a.value,
          b.value,
          c.value,
          d.value,
        ),
      )

      chisqStruct.constructFromFields(
        cb,
        r.region,
        FastSeq(
          EmitValue.present(primitive(cb.memoize(res(0)))),
          EmitValue.present(primitive(cb.memoize(res(1)))),
        ),
        deepCopy = false,
      )
    }

    registerSCode5(
      "contingency_table_test",
      TInt32,
      TInt32,
      TInt32,
      TInt32,
      TInt32,
      chisqStruct.virtualType,
      (_, _, _, _, _, _) => chisqStruct.sType,
    ) {
      case (
            r,
            cb,
            rt,
            a: SInt32Value,
            b: SInt32Value,
            c: SInt32Value,
            d: SInt32Value,
            mcc: SInt32Value,
            _,
          ) =>
        val res = cb.newLocal[Array[Double]](
          "contingency_table_test_res",
          Code.invokeScalaObject5[Int, Int, Int, Int, Int, Array[Double]](
            statsPackageClass,
            "contingencyTableTest",
            a.value,
            b.value,
            c.value,
            d.value,
            mcc.value,
          ),
        )

        chisqStruct.constructFromFields(
          cb,
          r.region,
          FastSeq(
            EmitValue.present(primitive(cb.memoize(res(0)))),
            EmitValue.present(primitive(cb.memoize(res(1)))),
          ),
          deepCopy = false,
        )
    }

    registerSCode4(
      "hardy_weinberg_test",
      TInt32,
      TInt32,
      TInt32,
      TBoolean,
      hweStruct.virtualType,
      (_, _, _, _, _) => hweStruct.sType,
    ) {
      case (
            r,
            cb,
            rt,
            nHomRef: SInt32Value,
            nHet: SInt32Value,
            nHomVar: SInt32Value,
            oneSided: SBooleanValue,
            _,
          ) =>
        val res = cb.newLocal[Array[Double]](
          "hardy_weinberg_test_res",
          Code.invokeScalaObject4[Int, Int, Int, Boolean, Array[Double]](
            statsPackageClass,
            "hardyWeinbergTest",
            nHomRef.value,
            nHet.value,
            nHomVar.value,
            oneSided.value,
          ),
        )

        hweStruct.constructFromFields(
          cb,
          r.region,
          FastSeq(
            EmitValue.present(primitive(cb.memoize(res(0)))),
            EmitValue.present(primitive(cb.memoize(res(1)))),
          ),
          deepCopy = false,
        )
    }
  }
}

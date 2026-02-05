package is.hail.expr.ir.functions

import is.hail.asm4s.Code
import is.hail.expr.ir.EmitValue
import is.hail.expr.ir.defs.{Cast, ErrorIDs}
import is.hail.stats._
import is.hail.types.physical.stypes.concrete._
import is.hail.types.physical.stypes.interfaces.primitive
import is.hail.types.physical.stypes.primitives._
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.utils.compat.immutable.ArraySeq

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

  def floor(x: Float): Float = math.floor(x.toDouble).toFloat

  def floor(x: Double): Double = math.floor(x)

  def ceil(x: Float): Float = math.ceil(x.toDouble).toFloat

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

  def pow(x: Int, y: Int): Double = math.pow(x.toDouble, y.toDouble)

  def pow(x: Long, y: Long): Double = math.pow(x.toDouble, y.toDouble)

  def pow(x: Float, y: Float): Double = math.pow(x.toDouble, y.toDouble)

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

  def floorDiv(x: Float, y: Float): Float = math.floor(x.toDouble / y).toFloat

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

  override def registerAll(): Unit = {
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

    registerScalaFunction("abs", ArraySeq(TInt32), TInt32, null)(mathPackageClass, "abs")
    registerScalaFunction("abs", ArraySeq(TInt64), TInt64, null)(mathPackageClass, "abs")
    registerScalaFunction("abs", ArraySeq(TFloat32), TFloat32, null)(mathPackageClass, "abs")
    registerScalaFunction("abs", ArraySeq(TFloat64), TFloat64, null)(mathPackageClass, "abs")

    registerScalaFunction("pow", ArraySeq(TInt32, TInt32), TFloat64, null)(thisClass, "pow")
    registerScalaFunction("pow", ArraySeq(TInt64, TInt64), TFloat64, null)(thisClass, "pow")
    registerScalaFunction("pow", ArraySeq(TFloat32, TFloat32), TFloat64, null)(thisClass, "pow")
    registerScalaFunction("pow", ArraySeq(TFloat64, TFloat64), TFloat64, null)(thisClass, "pow")

    registerScalaFunction("exp", ArraySeq(TFloat64), TFloat64, null)(mathPackageClass, "exp")
    registerScalaFunction("log10", ArraySeq(TFloat64), TFloat64, null)(mathPackageClass, "log10")
    registerScalaFunction("sqrt", ArraySeq(TFloat64), TFloat64, null)(mathPackageClass, "sqrt")
    registerScalaFunction("log", ArraySeq(TFloat64), TFloat64, null)(mathPackageClass, "log")
    registerScalaFunction("log", ArraySeq(TFloat64, TFloat64), TFloat64, null)(thisClass, "log")
    registerScalaFunction("log2", ArraySeq(TInt32), TInt32, null)(thisClass, "log2")
    registerScalaFunction("roundToNextPowerOf2", ArraySeq(TInt32), TInt32, null)(
      thisClass,
      "roundToNextPowerOf2",
    )
    registerScalaFunction("gamma", ArraySeq(TFloat64), TFloat64, null)(thisClass, "gamma")
    registerScalaFunction("binomTest", ArraySeq(TInt32, TInt32, TFloat64, TInt32), TFloat64, null)(
      statsPackageClass,
      "binomTest",
    )

    registerScalaFunction("dbeta", ArraySeq(TFloat64, TFloat64, TFloat64), TFloat64, null)(
      statsPackageClass,
      "dbeta",
    )

    registerScalaFunction("dnorm", ArraySeq(TFloat64), TFloat64, null)(statsPackageClass, "dnorm")
    registerScalaFunction(
      "dnorm",
      ArraySeq(TFloat64, TFloat64, TFloat64, TBoolean),
      TFloat64,
      null,
    )(
      statsPackageClass,
      "dnorm",
    )

    registerScalaFunction("pnorm", ArraySeq(TFloat64), TFloat64, null)(statsPackageClass, "pnorm")
    registerScalaFunction(
      "pnorm",
      ArraySeq(TFloat64, TFloat64, TFloat64, TBoolean, TBoolean),
      TFloat64,
      null,
    )(statsPackageClass, "pnorm")

    registerScalaFunction("qnorm", ArraySeq(TFloat64), TFloat64, null)(statsPackageClass, "qnorm")
    registerScalaFunction(
      "qnorm",
      ArraySeq(TFloat64, TFloat64, TFloat64, TBoolean, TBoolean),
      TFloat64,
      null,
    )(statsPackageClass, "qnorm")

    registerScalaFunction("pT", ArraySeq(TFloat64, TFloat64, TBoolean, TBoolean), TFloat64, null)(
      statsPackageClass,
      "pT",
    )
    registerScalaFunction(
      "pF",
      ArraySeq(TFloat64, TFloat64, TFloat64, TBoolean, TBoolean),
      TFloat64,
      null,
    )(statsPackageClass, "pF")

    registerScalaFunction("dpois", ArraySeq(TFloat64, TFloat64), TFloat64, null)(
      statsPackageClass,
      "dpois",
    )
    registerScalaFunction("dpois", ArraySeq(TFloat64, TFloat64, TBoolean), TFloat64, null)(
      statsPackageClass,
      "dpois",
    )

    registerScalaFunction("ppois", ArraySeq(TFloat64, TFloat64), TFloat64, null)(
      statsPackageClass,
      "ppois",
    )
    registerScalaFunction(
      "ppois",
      ArraySeq(TFloat64, TFloat64, TBoolean, TBoolean),
      TFloat64,
      null,
    )(
      statsPackageClass,
      "ppois",
    )

    registerScalaFunction("qpois", ArraySeq(TFloat64, TFloat64), TInt32, null)(
      statsPackageClass,
      "qpois",
    )
    registerScalaFunction("qpois", ArraySeq(TFloat64, TFloat64, TBoolean, TBoolean), TInt32, null)(
      statsPackageClass,
      "qpois",
    )

    registerScalaFunction(
      "dgamma",
      ArraySeq(TFloat64, TFloat64, TFloat64, TBoolean),
      TFloat64,
      null,
    )(
      statsPackageClass,
      "dgamma",
    )

    registerScalaFunction(
      "pgamma",
      ArraySeq(TFloat64, TFloat64, TFloat64, TBoolean, TBoolean),
      TFloat64,
      null,
    )(statsPackageClass, "pgamma")

    registerScalaFunction(
      "qgamma",
      ArraySeq(TFloat64, TFloat64, TFloat64, TBoolean, TBoolean),
      TFloat64,
      null,
    )(statsPackageClass, "qgamma")

    registerScalaFunction("dchisq", ArraySeq(TFloat64, TFloat64), TFloat64, null)(
      statsPackageClass,
      "dchisq",
    )
    registerScalaFunction("dchisq", ArraySeq(TFloat64, TFloat64, TBoolean), TFloat64, null)(
      statsPackageClass,
      "dchisq",
    )

    registerScalaFunction("dnchisq", ArraySeq(TFloat64, TFloat64, TFloat64), TFloat64, null)(
      statsPackageClass,
      "dnchisq",
    )
    registerScalaFunction(
      "dnchisq",
      ArraySeq(TFloat64, TFloat64, TFloat64, TBoolean),
      TFloat64,
      null,
    )(
      statsPackageClass,
      "dnchisq",
    )

    registerScalaFunction("pchisqtail", ArraySeq(TFloat64, TFloat64), TFloat64, null)(
      statsPackageClass,
      "pchisqtail",
    )
    registerScalaFunction(
      "pchisqtail",
      ArraySeq(TFloat64, TFloat64, TBoolean, TBoolean),
      TFloat64,
      null,
    )(statsPackageClass, "pchisqtail")

    registerScalaFunction("pnchisqtail", ArraySeq(TFloat64, TFloat64, TFloat64), TFloat64, null)(
      statsPackageClass,
      "pnchisqtail",
    )
    registerScalaFunction(
      "pnchisqtail",
      ArraySeq(TFloat64, TFloat64, TFloat64, TBoolean, TBoolean),
      TFloat64,
      null,
    )(statsPackageClass, "pnchisqtail")

    registerScalaFunction("qchisqtail", ArraySeq(TFloat64, TFloat64), TFloat64, null)(
      statsPackageClass,
      "qchisqtail",
    )
    registerScalaFunction(
      "qchisqtail",
      ArraySeq(TFloat64, TFloat64, TBoolean, TBoolean),
      TFloat64,
      null,
    )(statsPackageClass, "qchisqtail")

    registerScalaFunction("qnchisqtail", ArraySeq(TFloat64, TFloat64, TFloat64), TFloat64, null)(
      statsPackageClass,
      "qnchisqtail",
    )
    registerScalaFunction(
      "qnchisqtail",
      ArraySeq(TFloat64, TFloat64, TFloat64, TBoolean, TBoolean),
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
            _,
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
            Code.checkcast[IndexedSeq[Double]](svalueToJavaValue(cb, r, w)),
            Code.checkcast[IndexedSeq[Int]](svalueToJavaValue(cb, r, k)),
            Code.checkcast[IndexedSeq[Double]](svalueToJavaValue(cb, r, lam)),
            sigma.value,
            maxIterations.value,
            minAccuracy.value,
          ),
        )

        DaviesAlgorithm.pType.constructFromFields(
          cb,
          r,
          FastSeq(
            EmitValue.present(primitive(cb.memoize(res.invoke[Double]("value")))),
            EmitValue.present(primitive(cb.memoize(res.invoke[Int]("nIterations")))),
            EmitValue.present(primitive(cb.memoize(res.invoke[Boolean]("converged")))),
            EmitValue.present(primitive(cb.memoize(res.invoke[Int]("fault")))),
          ),
          deepCopy = false,
        )
    }

    registerScalaFunction(
      "phyper",
      ArraySeq(TInt32, TInt32, TInt32, TInt32, TBoolean),
      TFloat64,
      null,
    )(statsPackageClass, "phyper")

    registerScalaFunction("floor", ArraySeq(TFloat32), TFloat32, null)(thisClass, "floor")
    registerScalaFunction("floor", ArraySeq(TFloat64), TFloat64, null)(thisClass, "floor")

    registerScalaFunction("ceil", ArraySeq(TFloat32), TFloat32, null)(thisClass, "ceil")
    registerScalaFunction("ceil", ArraySeq(TFloat64), TFloat64, null)(thisClass, "ceil")

    registerScalaFunction("mod", ArraySeq(TInt32, TInt32), TInt32, null)(thisClass, "mod")
    registerScalaFunction("mod", ArraySeq(TInt64, TInt64), TInt64, null)(thisClass, "mod")
    registerScalaFunction("mod", ArraySeq(TFloat32, TFloat32), TFloat32, null)(thisClass, "mod")
    registerScalaFunction("mod", ArraySeq(TFloat64, TFloat64), TFloat64, null)(thisClass, "mod")

    registerJavaStaticFunction("isnan", ArraySeq(TFloat32), TBoolean, null)(jFloatClass, "isNaN")
    registerJavaStaticFunction("isnan", ArraySeq(TFloat64), TBoolean, null)(jDoubleClass, "isNaN")

    registerJavaStaticFunction("is_finite", ArraySeq(TFloat32), TBoolean, null)(
      jFloatClass,
      "isFinite",
    )
    registerJavaStaticFunction("is_finite", ArraySeq(TFloat64), TBoolean, null)(
      jDoubleClass,
      "isFinite",
    )

    registerJavaStaticFunction("is_infinite", ArraySeq(TFloat32), TBoolean, null)(
      jFloatClass,
      "isInfinite",
    )
    registerJavaStaticFunction("is_infinite", ArraySeq(TFloat64), TBoolean, null)(
      jDoubleClass,
      "isInfinite",
    )

    registerJavaStaticFunction("sign", ArraySeq(TInt32), TInt32, null)(jIntegerClass, "signum")
    registerScalaFunction("sign", ArraySeq(TInt64), TInt64, null)(mathPackageClass, "signum")
    registerJavaStaticFunction("sign", ArraySeq(TFloat32), TFloat32, null)(jMathClass, "signum")
    registerJavaStaticFunction("sign", ArraySeq(TFloat64), TFloat64, null)(jMathClass, "signum")

    registerScalaFunction(
      "approxEqual",
      ArraySeq(TFloat64, TFloat64, TFloat64, TBoolean, TBoolean),
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
    ) { case (r, cb, _, a: SInt32Value, b: SInt32Value, c: SInt32Value, d: SInt32Value, _) =>
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
        r,
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
    ) { case (r, cb, _, a: SInt32Value, b: SInt32Value, c: SInt32Value, d: SInt32Value, _) =>
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
        r,
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
            _,
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
          r,
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
            _,
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
          r,
          FastSeq(
            EmitValue.present(primitive(cb.memoize(res(0)))),
            EmitValue.present(primitive(cb.memoize(res(1)))),
          ),
          deepCopy = false,
        )
    }
  }
}

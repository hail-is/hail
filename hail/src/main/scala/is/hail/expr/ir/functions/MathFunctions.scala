package is.hail.expr.ir.functions

import is.hail.asm4s.Code
import is.hail.expr.ir._
import is.hail.stats._
import is.hail.types.physical.stypes.primitives.{SFloat64Code, SInt32Code}
import is.hail.types.physical.{PBoolean, PFloat32, PFloat64, PInt32, PInt64, PType}
import is.hail.types.virtual._
import is.hail.utils._
import org.apache.commons.math3.special.Gamma

object MathFunctions extends RegistryFunctions {
  def log(x: Double, b: Double): Double = math.log(x) / math.log(b)

  def gamma(x: Double): Double = Gamma.gamma(x)

  def floor(x: Float): Float = math.floor(x).toFloat

  def floor(x: Double): Double = math.floor(x)

  def ceil(x: Float): Float = math.ceil(x).toFloat

  def ceil(x: Double): Double = math.ceil(x)

  def mod(x: Int, y: Int): Int = {
    if (y == 0)
      fatal(s"$x % 0: modulo by zero")
    java.lang.Math.floorMod(x, y)
  }

  def mod(x: Long, y: Long): Long = {
    if (y == 0L)
      fatal(s"$x % 0: modulo by zero")
    java.lang.Math.floorMod(x, y)
  }

  def mod(x: Float, y: Float): Float = {
    if (y == 0.0)
      fatal(s"$x % 0: modulo by zero")
    val t = x % y
    if (t < 0) t + y else t
  }

  def mod(x: Double, y: Double): Double = {
    if (y == 0.0)
      fatal(s"$x % 0: modulo by zero")
    val t = x % y
    if (t < 0) t + y else t
  }

  def pow(x: Int, y: Int): Double = math.pow(x, y)

  def pow(x: Long, y: Long): Double = math.pow(x, y)

  def pow(x: Float, y: Float): Double = math.pow(x, y)

  def pow(x: Double, y: Double): Double = math.pow(x, y)

  def floorDiv(x: Int, y: Int): Int = {
    if (y == 0)
      fatal(s"$x // 0: integer division by zero")
    java.lang.Math.floorDiv(x, y)
  }


  def floorDiv(x: Long, y: Long): Long = {
    if (y == 0L)
      fatal(s"$x // 0: integer division by zero")
    java.lang.Math.floorDiv(x, y)
  }

  def floorDiv(x: Float, y: Float): Float = math.floor(x / y).toFloat

  def floorDiv(x: Double, y: Double): Double = math.floor(x / y)

  def approxEqual(x: Double, y: Double, tolerance: Double, absolute: Boolean, nanSame: Boolean): Boolean = {
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
    registerIR1("toInt32", tnum("T"), TInt32)((_, x) => Cast((x), TInt32))
    registerIR1("toInt64", tnum("T"), TInt64)((_, x) => Cast(x, TInt64))
    registerIR1("toFloat32", tnum("T"), TFloat32)((_, x) => Cast(x, TFloat32))
    registerIR1("toFloat64", tnum("T"), TFloat64)((_, x) => Cast(x, TFloat64))

    registerScalaFunction("abs", Array(TInt32), TInt32, (_: Type, _: Seq[PType]) => PInt32())(mathPackageClass, "abs")
    registerScalaFunction("abs", Array(TInt64), TInt64, (_: Type, _: Seq[PType]) => PInt64())(mathPackageClass, "abs")
    registerScalaFunction("abs", Array(TFloat32), TFloat32, (_: Type, _: Seq[PType]) => PFloat32())(mathPackageClass, "abs")
    registerScalaFunction("abs", Array(TFloat64), TFloat64, (_: Type, _: Seq[PType]) => PFloat64())(mathPackageClass, "abs")

    registerScalaFunction("pow", Array(TInt32, TInt32), TFloat64, (_: Type, _: Seq[PType]) => PFloat64())(thisClass, "pow")
    registerScalaFunction("pow", Array(TInt64, TInt64), TFloat64, (_: Type, _: Seq[PType]) => PFloat64())(thisClass, "pow")
    registerScalaFunction("pow", Array(TFloat32, TFloat32), TFloat64, (_: Type, _: Seq[PType]) => PFloat64())(thisClass, "pow")
    registerScalaFunction("pow", Array(TFloat64, TFloat64), TFloat64, (_: Type, _: Seq[PType]) => PFloat64())(thisClass, "pow")

    registerScalaFunction("exp", Array(TFloat64), TFloat64, (_: Type, _: Seq[PType]) => PFloat64())(mathPackageClass, "exp")
    registerScalaFunction("log10", Array(TFloat64), TFloat64, (_: Type, _: Seq[PType]) => PFloat64())(mathPackageClass, "log10")
    registerScalaFunction("sqrt", Array(TFloat64), TFloat64, (_: Type, _: Seq[PType]) => PFloat64())(mathPackageClass, "sqrt")
    registerScalaFunction("log", Array(TFloat64), TFloat64, (_: Type, _: Seq[PType]) => PFloat64())(mathPackageClass, "log")
    registerScalaFunction("log", Array(TFloat64, TFloat64), TFloat64, (_: Type, _: Seq[PType]) => PFloat64())(thisClass, "log")
    registerScalaFunction("gamma", Array(TFloat64), TFloat64, (_: Type, _: Seq[PType]) => PFloat64())(thisClass, "gamma")
    registerScalaFunction("binomTest", Array(TInt32, TInt32, TFloat64, TInt32), TFloat64, (_: Type, _: Seq[PType]) => PFloat64())(statsPackageClass, "binomTest")

    registerScalaFunction("dbeta", Array(TFloat64, TFloat64, TFloat64), TFloat64, (_: Type, _: Seq[PType]) => PFloat64())(statsPackageClass, "dbeta")

    registerScalaFunction("pnorm", Array(TFloat64), TFloat64, (_: Type, _: Seq[PType]) => PFloat64())(statsPackageClass, "pnorm")
    registerScalaFunction("qnorm", Array(TFloat64), TFloat64, (_: Type, _: Seq[PType]) => PFloat64())(statsPackageClass, "qnorm")

    registerScalaFunction("pT", Array(TFloat64, TFloat64, TBoolean, TBoolean), TFloat64, (_: Type, _: Seq[PType]) => PFloat64())(statsPackageClass, "pT")
    registerScalaFunction("pF", Array(TFloat64, TFloat64, TFloat64, TBoolean, TBoolean), TFloat64, (_: Type, _: Seq[PType]) => PFloat64())(statsPackageClass, "pF")

    registerScalaFunction("dpois", Array(TFloat64, TFloat64), TFloat64, (_: Type, _: Seq[PType]) => PFloat64())(statsPackageClass, "dpois")
    registerScalaFunction("dpois", Array(TFloat64, TFloat64, TBoolean), TFloat64, (_: Type, _: Seq[PType]) => PFloat64())(statsPackageClass, "dpois")

    registerScalaFunction("ppois", Array(TFloat64, TFloat64), TFloat64, (_: Type, _: Seq[PType]) => PFloat64())(statsPackageClass, "ppois")
    registerScalaFunction("ppois", Array(TFloat64, TFloat64, TBoolean, TBoolean), TFloat64, (_: Type, _: Seq[PType]) => PFloat64())(statsPackageClass, "ppois")

    registerScalaFunction("qpois", Array(TFloat64, TFloat64), TInt32, (_: Type, _: Seq[PType]) => PInt32())(statsPackageClass, "qpois")
    registerScalaFunction("qpois", Array(TFloat64, TFloat64, TBoolean, TBoolean), TInt32, (_: Type, _: Seq[PType]) => PInt32())(statsPackageClass, "qpois")

    registerScalaFunction("pchisqtail", Array(TFloat64, TFloat64), TFloat64, (_: Type, _: Seq[PType]) => PFloat64())(statsPackageClass, "chiSquaredTail")
    registerScalaFunction("pnchisqtail", Array(TFloat64, TFloat64, TFloat64), TFloat64, (_: Type, _: Seq[PType]) => PFloat64())(statsPackageClass, "nonCentralChiSquaredTail")
    registerScalaFunction("qchisqtail", Array(TFloat64, TFloat64), TFloat64, (_: Type, _: Seq[PType]) => PFloat64())(statsPackageClass, "inverseChiSquaredTail")

    registerScalaFunction("floor", Array(TFloat32), TFloat32, (_: Type, _: Seq[PType]) => PFloat32())(thisClass, "floor")
    registerScalaFunction("floor", Array(TFloat64), TFloat64, (_: Type, _: Seq[PType]) => PFloat64())(thisClass, "floor")

    registerScalaFunction("ceil", Array(TFloat32), TFloat32, (_: Type, _: Seq[PType]) => PFloat32())(thisClass, "ceil")
    registerScalaFunction("ceil", Array(TFloat64), TFloat64, (_: Type, _: Seq[PType]) => PFloat64())(thisClass, "ceil")

    registerScalaFunction("mod", Array(TInt32, TInt32), TInt32, (_: Type, _: Seq[PType]) => PInt32())(thisClass, "mod")
    registerScalaFunction("mod", Array(TInt64, TInt64), TInt64, (_: Type, _: Seq[PType]) => PInt64())(thisClass, "mod")
    registerScalaFunction("mod", Array(TFloat32, TFloat32), TFloat32, (_: Type, _: Seq[PType]) => PFloat32())(thisClass, "mod")
    registerScalaFunction("mod", Array(TFloat64, TFloat64), TFloat64, (_: Type, _: Seq[PType]) => PFloat64())(thisClass, "mod")

    registerJavaStaticFunction("isnan", Array(TFloat32), TBoolean, (_: Type, _: Seq[PType]) => PBoolean())(jFloatClass, "isNaN")
    registerJavaStaticFunction("isnan", Array(TFloat64), TBoolean, (_: Type, _: Seq[PType]) => PBoolean())(jDoubleClass, "isNaN")

    registerJavaStaticFunction("is_finite", Array(TFloat32), TBoolean, (_: Type, _: Seq[PType]) => PBoolean())(jFloatClass, "isFinite")
    registerJavaStaticFunction("is_finite", Array(TFloat64), TBoolean, (_: Type, _: Seq[PType]) => PBoolean())(jDoubleClass, "isFinite")

    registerJavaStaticFunction("is_infinite", Array(TFloat32), TBoolean, (_: Type, _: Seq[PType]) => PBoolean())(jFloatClass, "isInfinite")
    registerJavaStaticFunction("is_infinite", Array(TFloat64), TBoolean, (_: Type, _: Seq[PType]) => PBoolean())(jDoubleClass, "isInfinite")

    registerJavaStaticFunction("sign", Array(TInt32), TInt32, (_: Type, _: Seq[PType]) => PInt32())(jIntegerClass, "signum")
    registerScalaFunction("sign", Array(TInt64), TInt64, (_: Type, _: Seq[PType]) => PInt64())(mathPackageClass, "signum")
    registerJavaStaticFunction("sign", Array(TFloat32), TFloat32, (_: Type, _: Seq[PType]) => PFloat32())(jMathClass, "signum")
    registerJavaStaticFunction("sign", Array(TFloat64), TFloat64, (_: Type, _: Seq[PType]) => PFloat64())(jMathClass, "signum")

    registerScalaFunction("approxEqual", Array(TFloat64, TFloat64, TFloat64, TBoolean, TBoolean), TBoolean, (_: Type, _: Seq[PType]) => PBoolean())(thisClass, "approxEqual")

    registerWrappedScalaFunction1("entropy", TString, TFloat64, (_: Type, _: PType) => PFloat64())(thisClass, "irentropy")

    registerPCode4("fisher_exact_test", TInt32, TInt32, TInt32, TInt32, fetStruct.virtualType,
      (_, _, _, _, _) => fetStruct
    ) { case (r, cb, rt, a: SInt32Code, b: SInt32Code, c: SInt32Code, d: SInt32Code) =>
      val res = cb.newLocal[Array[Double]]("fisher_exact_test_res",
        Code.invokeScalaObject4[Int, Int, Int, Int, Array[Double]](statsPackageClass, "fisherExactTest",
          a.intCode(cb),
          b.intCode(cb),
          c.intCode(cb),
          d.intCode(cb)))

      fetStruct.constructFromFields(cb, r.region, FastIndexedSeq(
        IEmitCode.present(cb, SFloat64Code(res(0))).memoize(cb, "fisher_exact_test_res0"),
        IEmitCode.present(cb, SFloat64Code(res(1))).memoize(cb, "fisher_exact_test_res1"),
        IEmitCode.present(cb, SFloat64Code(res(2))).memoize(cb, "fisher_exact_test_res2"),
        IEmitCode.present(cb, SFloat64Code(res(3))).memoize(cb, "fisher_exact_test_res3")
      ), deepCopy = false)
    }

    registerPCode4("chi_squared_test", TInt32, TInt32, TInt32, TInt32, chisqStruct.virtualType,
      (_, _, _, _, _) => chisqStruct
    ) { case (r, cb, rt, a: SInt32Code, b: SInt32Code, c: SInt32Code, d: SInt32Code) =>
      val res = cb.newLocal[Array[Double]]("chi_squared_test_res",
        Code.invokeScalaObject4[Int, Int, Int, Int, Array[Double]](statsPackageClass, "chiSquaredTest",
          a.intCode(cb),
          b.intCode(cb),
          c.intCode(cb),
          d.intCode(cb)))

      chisqStruct.constructFromFields(cb, r.region, FastIndexedSeq(
        IEmitCode.present(cb, SFloat64Code(res(0))).memoize(cb, "chi_squared_test_res0"),
        IEmitCode.present(cb, SFloat64Code(res(1))).memoize(cb, "chi_squared_test_res1")
      ), deepCopy = false)
    }

    registerPCode5("contingency_table_test", TInt32, TInt32, TInt32, TInt32, TInt32, chisqStruct.virtualType,
      (_, _, _, _, _, _) => chisqStruct
    ) { case (r, cb, rt, a: SInt32Code, b: SInt32Code, c: SInt32Code, d: SInt32Code, mcc: SInt32Code) =>
      val res = cb.newLocal[Array[Double]]("contingency_table_test_res",
        Code.invokeScalaObject5[Int, Int, Int, Int, Int, Array[Double]](statsPackageClass, "contingencyTableTest",
          a.intCode(cb),
          b.intCode(cb),
          c.intCode(cb),
          d.intCode(cb),
          mcc.intCode(cb)))

      chisqStruct.constructFromFields(cb, r.region, FastIndexedSeq(
        IEmitCode.present(cb, SFloat64Code(res(0))).memoize(cb, "contingency_table_test_res0"),
        IEmitCode.present(cb, SFloat64Code(res(1))).memoize(cb, "contingency_table_test_res1")
      ), deepCopy = false)
    }

    registerPCode3("hardy_weinberg_test", TInt32, TInt32, TInt32, hweStruct.virtualType,
      (_, _, _, _) => hweStruct
    ) { case (r, cb, rt, nHomRef: SInt32Code, nHet: SInt32Code, nHomVar: SInt32Code) =>
      val res = cb.newLocal[Array[Double]]("hardy_weinberg_test_res",
        Code.invokeScalaObject3[Int, Int, Int, Array[Double]](statsPackageClass, "hardyWeinbergTest",
          nHomRef.intCode(cb),
          nHet.intCode(cb),
          nHomVar.intCode(cb)))

      hweStruct.constructFromFields(cb, r.region, FastIndexedSeq(
        IEmitCode.present(cb, SFloat64Code(res(0))).memoize(cb, "hardy_weinberg_test_res0"),
        IEmitCode.present(cb, SFloat64Code(res(1))).memoize(cb, "hardy_weinberg_test_res1")
      ), deepCopy = false)
    }
  }
}

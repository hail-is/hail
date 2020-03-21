package is.hail.expr.ir.functions

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s.{AsmFunction3, Code}
import is.hail.expr.ir._
import is.hail.expr.types._
import org.apache.commons.math3.special.Gamma
import is.hail.stats._
import is.hail.utils._
import is.hail.asm4s
import is.hail.expr.types.physical.PType
import is.hail.expr.types.virtual._

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
    registerIR("toInt32", tnum("T"), TInt32)(x => Cast(x, TInt32))
    registerIR("toInt64", tnum("T"), TInt64)(x => Cast(x, TInt64))
    registerIR("toFloat32", tnum("T"), TFloat32)(x => Cast(x, TFloat32))
    registerIR("toFloat64", tnum("T"), TFloat64)(x => Cast(x, TFloat64))
    
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
    registerScalaFunction("gamma", Array(TFloat64), TFloat64, null)(thisClass, "gamma")
    registerScalaFunction("binomTest", Array(TInt32, TInt32, TFloat64, TInt32), TFloat64, null)(statsPackageClass, "binomTest")

    registerScalaFunction("dbeta", Array(TFloat64, TFloat64, TFloat64), TFloat64, null)(statsPackageClass, "dbeta")

    registerScalaFunction("pnorm", Array(TFloat64), TFloat64, null)(statsPackageClass, "pnorm")
    registerScalaFunction("qnorm", Array(TFloat64), TFloat64, null)(statsPackageClass, "qnorm")

    registerScalaFunction("pT", Array(TFloat64, TFloat64, TBoolean, TBoolean), TFloat64, null)(statsPackageClass, "pT")
    registerScalaFunction("pF", Array(TFloat64, TFloat64, TFloat64, TBoolean, TBoolean), TFloat64, null)(statsPackageClass, "pF")

    registerScalaFunction("dpois", Array(TFloat64, TFloat64), TFloat64, null)(statsPackageClass, "dpois")
    registerScalaFunction("dpois", Array(TFloat64, TFloat64, TBoolean), TFloat64, null)(statsPackageClass, "dpois")

    registerScalaFunction("ppois", Array(TFloat64, TFloat64), TFloat64, null)(statsPackageClass, "ppois")
    registerScalaFunction("ppois", Array(TFloat64, TFloat64, TBoolean, TBoolean), TFloat64, null)(statsPackageClass, "ppois")

    registerScalaFunction("qpois", Array(TFloat64, TFloat64), TInt32, null)(statsPackageClass, "qpois")
    registerScalaFunction("qpois", Array(TFloat64, TFloat64, TBoolean, TBoolean), TInt32, null)(statsPackageClass, "qpois")

    registerScalaFunction("pchisqtail", Array(TFloat64, TFloat64), TFloat64, null)(statsPackageClass, "chiSquaredTail")
    registerScalaFunction("qchisqtail", Array(TFloat64, TFloat64), TFloat64, null)(statsPackageClass, "inverseChiSquaredTail")

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

    registerJavaStaticFunction("is_finite", Array(TFloat32), TBoolean, null)(jFloatClass, "isFinite")
    registerJavaStaticFunction("is_finite", Array(TFloat64), TBoolean, null)(jDoubleClass, "isFinite")

    registerJavaStaticFunction("is_infinite", Array(TFloat32), TBoolean, null)(jFloatClass, "isInfinite")
    registerJavaStaticFunction("is_infinite", Array(TFloat64), TBoolean, null)(jDoubleClass, "isInfinite")

    registerJavaStaticFunction("sign", Array(TInt32), TInt32, null)(jIntegerClass, "signum")
    registerScalaFunction("sign", Array(TInt64), TInt64, null)(mathPackageClass, "signum")
    registerJavaStaticFunction("sign", Array(TFloat32), TFloat32, null)(jMathClass, "signum")
    registerJavaStaticFunction("sign", Array(TFloat64), TFloat64, null)(jMathClass, "signum")
    
    registerScalaFunction("approxEqual", Array(TFloat64, TFloat64, TFloat64, TBoolean, TBoolean), TBoolean, null)(thisClass, "approxEqual")

    registerWrappedScalaFunction("entropy", TString, TFloat64, null)(thisClass, "irentropy")

    registerCode("fisher_exact_test", TInt32, TInt32, TInt32, TInt32, fetStruct.virtualType,
      (_, _, _, _) => fetStruct
    ){ case (r, rt, (at, a), (bt, b), (ct, c), (dt, d)) =>
      val res = r.mb.newLocal[Array[Double]]()
      val srvb = new StagedRegionValueBuilder(r, rt)
      Code(Code(FastIndexedSeq(
        res := Code.invokeScalaObject[Int, Int, Int, Int, Array[Double]](statsPackageClass, "fisherExactTest", a, b, c, d),
        srvb.start(),
        srvb.addDouble(res(0)),
        srvb.advance(),
        srvb.addDouble(res(1)),
        srvb.advance(),
        srvb.addDouble(res(2)),
        srvb.advance(),
        srvb.addDouble(res(3)),
        srvb.advance())),
        srvb.offset)
    }
    
    registerCode("chi_squared_test", TInt32, TInt32, TInt32, TInt32, chisqStruct.virtualType,
      (_, _, _, _) => chisqStruct
    ){ case (r, rt, (at, a), (bt, b), (ct, c), (dt, d))  =>
      val res = r.mb.newLocal[Array[Double]]()
      val srvb = new StagedRegionValueBuilder(r, rt)
      Code(
        res := Code.invokeScalaObject[Int, Int, Int, Int, Array[Double]](statsPackageClass, "chiSquaredTest", a, b, c, d),
        srvb.start(),
        srvb.addDouble(res(0)),
        srvb.advance(),
        srvb.addDouble(res(1)),
        srvb.advance(),
        srvb.offset
      )
    }

    registerCode("contingency_table_test", TInt32, TInt32, TInt32, TInt32, TInt32, chisqStruct.virtualType,
      (_, _, _, _, _) => chisqStruct
    ){ case (r, rt, (at, a), (bt, b), (ct, c), (dt, d), (mccT, min_cell_count)) =>
      val res = r.mb.newLocal[Array[Double]]()
      val srvb = new StagedRegionValueBuilder(r, rt)
      Code(
        res := Code.invokeScalaObject[Int, Int, Int, Int, Int, Array[Double]](statsPackageClass, "contingencyTableTest", a, b, c, d, min_cell_count),
        srvb.start(),
        srvb.addDouble(res(0)),
        srvb.advance(),
        srvb.addDouble(res(1)),
        srvb.advance(),
        srvb.offset
      )
    }

    registerCode("hardy_weinberg_test", TInt32, TInt32, TInt32,
      hweStruct.virtualType, (_, _, _) => hweStruct) { case (r, rt, (nhrT, nHomRef), (nhT, nHet), (nhvT, nHomVar)) =>
      val res = r.mb.newLocal[Array[Double]]()
      val srvb = new StagedRegionValueBuilder(r, rt)
      Code(
        res := Code.invokeScalaObject[Int, Int, Int, Array[Double]](statsPackageClass, "hardyWeinbergTest", nHomRef, nHet, nHomVar),
        srvb.start(),
        srvb.addDouble(res(0)),
        srvb.advance(),
        srvb.addDouble(res(1)),
        srvb.advance(),
        srvb.offset
      )
    }
  }
}

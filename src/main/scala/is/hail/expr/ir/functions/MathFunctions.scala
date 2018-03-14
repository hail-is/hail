package is.hail.expr.ir.functions

import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.stats
import org.apache.commons.math3.special.Gamma

object MathFunctions extends RegistryFunctions {
  // FIXME can't figure out how to get the instance of a package object
  def exp(x: Double): Double = math.exp(x)

  def log10(x: Double): Double = math.log10(x)

  def sqrt(x: Double): Double = math.sqrt(x)

  def log(x: Double): Double = math.log(x)

  def log(x: Double, b: Double): Double = math.log(x) / math.log(b)

  def pow(b: Double, x: Double): Double = math.pow(b, x)

  def gamma(x: Double): Double = Gamma.gamma(x)

  def binomTest(nSuccess: Int, n: Int, p: Double, alternative: String): Double = stats.binomTest(nSuccess, n, p, alternative)

  def dbeta(x: Double, a: Double, b: Double): Double = stats.dbeta(x, a, b)

  def rnorm(mean: Double, sd: Double): Double = mean + sd * scala.util.Random.nextGaussian()

  def pnorm(x: Double): Double = stats.pnorm(x)

  def qnorm(p: Double): Double = stats.qnorm(p)

  def rpois(lambda: Double): Double = stats.rpois(lambda)

  def floor(x: Float): Float = math.floor(x).toFloat

  def floor(x: Double): Double = math.floor(x)

  def ceil(x: Float): Float = math.ceil(x).toFloat

  def ceil(x: Double): Double = math.ceil(x)

  def floorMod(x: Int, y: Int): Int = java.lang.Math.floorMod(x, y)

  def floorMod(x: Long, y: Long): Long = java.lang.Math.floorMod(x, y)

  def floorDiv(x: Int, y: Int): Int = java.lang.Math.floorDiv(x, y)

  def floorDiv(x: Long, y: Long): Long = java.lang.Math.floorDiv(x, y)

  def floorDiv(x: Float, y: Float): Float = math.floor(x / y).toFloat

  def floorDiv(x: Double, y: Double): Double = math.floor(x / y)

  def isnan(d: Double): Boolean = d.isNaN

  def registerAll() {
    // numeric conversions
    registerIR("toInt32", tnum("T"))(x => Cast(x, TInt32()))
    registerIR("toInt64", tnum("T"))(x => Cast(x, TInt64()))
    registerIR("toFloat32", tnum("T"))(x => Cast(x, TFloat32()))
    registerIR("toFloat64", tnum("T"))(x => Cast(x, TFloat64()))

    registerScalaFunction("exp", TFloat64(), TFloat64())(this, "exp")
    registerScalaFunction("log10", TFloat64(), TFloat64())(this, "log10")
    registerScalaFunction("sqrt", TFloat64(), TFloat64())(this, "sqrt")
    registerScalaFunction("log", TFloat64(), TFloat64())(this, "log")
    registerScalaFunction("log", TFloat64(), TFloat64(), TFloat64())(this, "log")
    registerScalaFunction("pow", TFloat64(), TFloat64(), TFloat64())(this, "pow")
    registerScalaFunction("**", TFloat64(), TFloat64(), TFloat64())(this, "pow")
    registerScalaFunction("gamma", TFloat64(), TFloat64())(this, "gamma")

    registerScalaFunction("binomTest", TInt32(), TInt32(), TFloat64(), TString(), TFloat64())(this, "binomTest")

    registerScalaFunction("dbeta", TFloat64(), TFloat64(), TFloat64(), TFloat64())(this, "dbeta")

    registerScalaFunction("rnorm", TFloat64(), TFloat64(), TFloat64())(this, "rnorm")

    registerScalaFunction("pnorm", TFloat64(), TFloat64())(this, "pnorm")
    registerScalaFunction("qnorm", TFloat64(), TFloat64())(this, "qnorm")

    registerScalaFunction("rpois", TFloat64(), TFloat64())(this, "rpois")
    // other rpois returns an array

    registerScalaFunction("floor", TFloat32(), TFloat32())(this, "floor")
    registerScalaFunction("floor", TFloat64(), TFloat64())(this, "floor")

    registerScalaFunction("ceil", TFloat32(), TFloat32())(this, "ceil")
    registerScalaFunction("ceil", TFloat64(), TFloat64())(this, "ceil")

    registerScalaFunction("//", TInt32(), TInt32(), TInt32())(this, "floorDiv")
    registerScalaFunction("//", TInt64(), TInt64(), TInt64())(this, "floorDiv")
    registerScalaFunction("//", TFloat32(), TFloat32(), TFloat32())(this, "floorDiv")
    registerScalaFunction("//", TFloat64(), TFloat64(), TFloat64())(this, "floorDiv")

    registerScalaFunction("%", TInt32(), TInt32(), TInt32())(this, "floorMod")
    registerScalaFunction("%", TInt64(), TInt32(), TInt64())(this, "floorMod")

    registerScalaFunction("isnan", TFloat64(), TBoolean())(this, "isnan")
  }
}

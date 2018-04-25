package is.hail.expr.ir.functions

import is.hail.asm4s.Code
import is.hail.expr.ir._
import is.hail.expr.types._
import org.apache.commons.math3.special.Gamma
import is.hail.expr.ir.coerce

object MathFunctions extends RegistryFunctions {

  def log(x: Double, b: Double): Double = math.log(x) / math.log(b)

  def gamma(x: Double): Double = Gamma.gamma(x)

  def rnorm(mean: Double, sd: Double): Double = mean + sd * scala.util.Random.nextGaussian()

  def floor(x: Float): Float = math.floor(x).toFloat

  def floor(x: Double): Double = math.floor(x)

  def ceil(x: Float): Float = math.ceil(x).toFloat

  def ceil(x: Double): Double = math.ceil(x)

  def floorDiv(x: Int, y: Int): Int = java.lang.Math.floorDiv(x, y)

  def floorDiv(x: Long, y: Long): Long = java.lang.Math.floorDiv(x, y)

  def floorDiv(x: Float, y: Float): Float = math.floor(x / y).toFloat

  def floorDiv(x: Double, y: Double): Double = math.floor(x / y)

  def isnan(d: Double): Boolean = d.isNaN

  def pcoin(p: Double): Boolean = math.random < p

  def runif(min: Double, max: Double): Double = min + (max - min) * math.random

  def registerAll() {
    val thisClass = getClass
    val mathPackageClass = Class.forName("scala.math.package$")
    val statsPackageClass = Class.forName("is.hail.stats.package$")
    val jMathClass = classOf[java.lang.Math]

    // numeric conversions
    registerIR("toInt32", tnum("T"))(x => Cast(x, TInt32()))
    registerIR("toInt64", tnum("T"))(x => Cast(x, TInt64()))
    registerIR("toFloat32", tnum("T"))(x => Cast(x, TFloat32()))
    registerIR("toFloat64", tnum("T"))(x => Cast(x, TFloat64()))

    registerScalaFunction("exp", TFloat64(), TFloat64())(mathPackageClass, "exp")
    registerScalaFunction("log10", TFloat64(), TFloat64())(mathPackageClass, "log10")
    registerScalaFunction("sqrt", TFloat64(), TFloat64())(mathPackageClass, "sqrt")
    registerScalaFunction("log", TFloat64(), TFloat64())(mathPackageClass, "log")
    registerScalaFunction("log", TFloat64(), TFloat64(), TFloat64())(thisClass, "log")
    registerScalaFunction("pow", TFloat64(), TFloat64(), TFloat64())(mathPackageClass, "pow")
    registerScalaFunction("**", TFloat64(), TFloat64(), TFloat64())(mathPackageClass, "pow")
    registerScalaFunction("gamma", TFloat64(), TFloat64())(thisClass, "gamma")

    // FIXME: This needs to do string conversion.
    // registerScalaFunction("binomTest", TInt32(), TInt32(), TFloat64(), TString(), TFloat64())(statsPackageClass, "binomTest")

    registerScalaFunction("dbeta", TFloat64(), TFloat64(), TFloat64(), TFloat64())(statsPackageClass, "dbeta")

    registerScalaFunction("rnorm", TFloat64(), TFloat64(), TFloat64())(thisClass, "rnorm", isDeterministic = false)

    registerScalaFunction("pnorm", TFloat64(), TFloat64())(statsPackageClass, "pnorm")
    registerScalaFunction("qnorm", TFloat64(), TFloat64())(statsPackageClass, "qnorm")

    registerScalaFunction("rpois", TFloat64(), TFloat64())(statsPackageClass, "rpois", isDeterministic = false)
    // other rpois returns an array

    registerScalaFunction("dpois", TFloat64(), TFloat64(), TFloat64())(statsPackageClass, "dpois")
    registerScalaFunction("dpois", TFloat64(), TFloat64(), TBoolean(), TFloat64())(statsPackageClass, "dpois")

    registerScalaFunction("ppois", TFloat64(), TFloat64(), TFloat64())(statsPackageClass, "ppois")
    registerScalaFunction("ppois", TFloat64(), TFloat64(), TBoolean(), TBoolean(), TFloat64())(statsPackageClass, "ppois")

    registerScalaFunction("qpois", TFloat64(), TFloat64(), TInt32())(statsPackageClass, "qpois")
    registerScalaFunction("qpois", TFloat64(), TFloat64(), TBoolean(), TBoolean(), TInt32())(statsPackageClass, "qpois")

    registerScalaFunction("pchisqtail", TFloat64(), TFloat64(), TFloat64())(statsPackageClass, "chiSquaredTail")
    registerScalaFunction("qchisqtail", TFloat64(), TFloat64(), TFloat64())(statsPackageClass, "inverseChiSquaredTail")

    registerScalaFunction("pcoin", TFloat64(), TBoolean())(thisClass, "pcoin", isDeterministic = false)
    registerScalaFunction("runif", TFloat64(), TFloat64(), TFloat64())(thisClass, "runif", isDeterministic = false)

    registerScalaFunction("floor", TFloat32(), TFloat32())(thisClass, "floor")
    registerScalaFunction("floor", TFloat64(), TFloat64())(thisClass, "floor")

    registerScalaFunction("ceil", TFloat32(), TFloat32())(thisClass, "ceil")
    registerScalaFunction("ceil", TFloat64(), TFloat64())(thisClass, "ceil")

    registerScalaFunction("//", TInt32(), TInt32(), TInt32())(thisClass, "floorDiv")
    registerScalaFunction("//", TInt64(), TInt64(), TInt64())(thisClass, "floorDiv")
    registerScalaFunction("//", TFloat32(), TFloat32(), TFloat32())(thisClass, "floorDiv")
    registerScalaFunction("//", TFloat64(), TFloat64(), TFloat64())(thisClass, "floorDiv")

    registerJavaStaticFunction("%", TInt32(), TInt32(), TInt32())(jMathClass, "floorMod")
    registerJavaStaticFunction("%", TInt64(), TInt32(), TInt64())(jMathClass, "floorMod")

    registerScalaFunction("isnan", TFloat64(), TBoolean())(thisClass, "isnan")

    registerCodeWithMissingness("&&", TBoolean(), TBoolean(), TBoolean()) { (_, l, r) =>
      EmitTriplet(
        Code(l.setup, r.setup),
        l.m || (coerce[Boolean](l.v) && r.m),
        coerce[Boolean](l.v) && coerce[Boolean](r.v)
      )
    }

    registerCodeWithMissingness("||", TBoolean(), TBoolean(), TBoolean()) { (_, l, r) =>
      EmitTriplet(
        Code(l.setup, r.setup),
        l.m || (!coerce[Boolean](l.v) && r.m),
        coerce[Boolean](l.v) || coerce[Boolean](r.v)
      )
    }
  }
}

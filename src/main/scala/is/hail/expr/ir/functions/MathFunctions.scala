package is.hail.expr.ir.functions

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s.{AsmFunction3, Code}
import is.hail.expr.ir._
import is.hail.expr.types._
import org.apache.commons.math3.special.Gamma
import org.apache.commons.math3.random.RandomDataGenerator
import is.hail.stats.{chisqStruct, entropy, fetStruct, hweStruct, uniroot}
import is.hail.utils.fatal


class IRRandomness(seed: Long, partitionIndex: Int) {
  private[this] val combinedSeed = seed ^ java.lang.Math.floorMod(partitionIndex * 11399, 17863)

  val random: RandomDataGenerator = new RandomDataGenerator()
  random.reSeed(combinedSeed)

  def runif(min: Double, max: Double): Double = random.nextUniform(min, max, true)

  def rcoin(p: Double): Boolean = runif(0, 1) < p

  def rpois(lambda: Double): Double = random.nextPoisson(lambda)

  def rnorm(mean: Double, sd: Double): Double = random.nextGaussian(mean, sd)
}

object RandomSeededFunctions extends RegistryFunctions {

  def registerSeeded(mname: String, aTypes: Array[Type], rType: Type)(impl: (EmitMethodBuilder, Long, Array[Code[_]]) => Code[_]) {
    IRFunctionRegistry.addIRFunction(new SeededIRFunction {
      val isDeterministic: Boolean = false

      override val name: String = mname

      override val argTypes: Seq[Type] = aTypes :+ TInt64()

      override val returnType: Type = rType

      override def applySeeded(seed: Long, mb: EmitMethodBuilder, args: Code[_]*): Code[_] = impl(mb, seed, args.toArray)
    })
  }

  def registerSeeded[A1](mname: String, arg1: Type, rType: Type)(impl: (EmitMethodBuilder, Long, Code[A1]) => Code[_]): Unit =
    registerSeeded(mname, Array(arg1), rType) { (emb, seed, array) =>
      (emb: @unchecked, array: @unchecked) match {
        case (mb, Array(a1: Code[A1] @unchecked)) => impl(mb, seed, a1)
      }
    }

  def registerSeeded[A1, A2](mname: String, arg1: Type, arg2: Type, rType: Type)(impl: (EmitMethodBuilder, Long, Code[A1], Code[A2]) => Code[_]): Unit =
    registerSeeded(mname, Array(arg1, arg2), rType) { (emb, seed, array) =>
      (emb: @unchecked, array: @unchecked) match {
        case (mb, Array(a1: Code[A1] @unchecked, a2: Code[A2] @unchecked)) => impl(mb, seed, a1, a2)
      }
    }

  def registerAll() {
    registerSeeded("runif_seeded", TFloat64(), TFloat64(), TFloat64()) { case (mb, seed, min, max) =>
      mb.getRNG(seed).invoke[Double, Double, Double]("runif", min, max)
    }

    registerSeeded("rnorm_seeded", TFloat64(), TFloat64(), TFloat64()) { case (mb, seed, mean, sd) =>
      mb.getRNG(seed).invoke[Double, Double, Double]("rnorm", mean, sd)
    }

    registerSeeded("pcoin_seeded", TFloat64(), TBoolean()) { case (mb, seed, p) =>
      mb.getRNG(seed).invoke[Double, Boolean]("rcoin", p)
    }

    registerSeeded("rpois_seeded", TFloat64(), TFloat64()) { case (mb, seed, lambda) =>
      mb.getRNG(seed).invoke[Double, Double]("rpois", lambda)
    }

    registerSeeded("rpois_seeded", TInt32(), TFloat64(), TArray(TFloat64())) { case (mb, seed, n, lambda) =>
      val length = mb.newLocal[Int]
      val srvb = new StagedRegionValueBuilder(mb, TArray(TFloat64()))
      Code(
        length := n,
        srvb.start(n),
        Code.whileLoop(srvb.arrayIdx < length,
          srvb.addDouble(mb.getRNG(seed).invoke[Double, Double]("rpois", lambda)),
          srvb.advance()
        ),
        srvb.offset)
    }
  }
}

object MathFunctions extends RegistryFunctions {
  def log(x: Double, b: Double): Double = math.log(x) / math.log(b)

  def gamma(x: Double): Double = Gamma.gamma(x)

  def floor(x: Float): Float = math.floor(x).toFloat

  def floor(x: Double): Double = math.floor(x)

  def ceil(x: Float): Float = math.ceil(x).toFloat

  def ceil(x: Double): Double = math.ceil(x)

  def mod(x: Float, y: Float): Float = {
    val t = x % y
    if (t < 0) t + y else t
  }
  def mod(x: Double, y: Double): Double = {
    val t = x % y
    if (t < 0) t + y else t
  }

  def pow(x: Int, y: Int): Double = math.pow(x, y)
  def pow(x: Long, y: Long): Double = math.pow(x, y)
  def pow(x: Float, y: Float): Double = math.pow(x, y)
  def pow(x: Double, y: Double): Double = math.pow(x, y)

  def floorDiv(x: Int, y: Int): Int = java.lang.Math.floorDiv(x, y)

  def floorDiv(x: Long, y: Long): Long = java.lang.Math.floorDiv(x, y)

  def floorDiv(x: Float, y: Float): Float = math.floor(x / y).toFloat

  def floorDiv(x: Double, y: Double): Double = math.floor(x / y)

  def iruniroot(region: Region, irf: AsmFunction3[Region, Double, Boolean, Double], min: Double, max: Double): java.lang.Double = {
    val f: Double => Double = irf(region, _, false)
    if (!(min < max))
      fatal(s"min must be less than max in call to uniroot, got: min $min, max $max")

    val fmin = f(min)
    val fmax = f(max)

    if (fmin * fmax > 0.0)
      fatal(s"sign of endpoints must have opposite signs, got: f(min) = $fmin, f(max) = $fmax")

    val r = uniroot(f, min, max)
    if (r.isEmpty)
      null
    else
      r.get
  }
  
  def irentropy(s: String): Double = entropy(s)

  def registerAll() {
    val thisClass = getClass
    val mathPackageClass = Class.forName("scala.math.package$")
    val statsPackageClass = Class.forName("is.hail.stats.package$")
    val jMathClass = classOf[java.lang.Math]
    val jIntegerClass = classOf[java.lang.Integer]
    val jFloatClass = classOf[java.lang.Float]
    val jDoubleClass = classOf[java.lang.Double]    

    // numeric conversions
    registerIR("toInt32", tnum("T"))(x => Cast(x, TInt32()))
    registerIR("toInt64", tnum("T"))(x => Cast(x, TInt64()))
    registerIR("toFloat32", tnum("T"))(x => Cast(x, TFloat32()))
    registerIR("toFloat64", tnum("T"))(x => Cast(x, TFloat64()))
    
    registerScalaFunction("abs", TInt32(), TInt32())(mathPackageClass, "abs")
    registerScalaFunction("abs", TInt64(), TInt64())(mathPackageClass, "abs")
    registerScalaFunction("abs", TFloat32(), TFloat32())(mathPackageClass, "abs")
    registerScalaFunction("abs", TFloat64(), TFloat64())(mathPackageClass, "abs")

    registerScalaFunction("**", TInt32(), TInt32(), TFloat64())(thisClass, "pow")
    registerScalaFunction("**", TInt64(), TInt64(), TFloat64())(thisClass, "pow")
    registerScalaFunction("**", TFloat32(), TFloat32(), TFloat64())(thisClass, "pow")
    registerScalaFunction("**", TFloat64(), TFloat64(), TFloat64())(thisClass, "pow")

    registerScalaFunction("exp", TFloat64(), TFloat64())(mathPackageClass, "exp")
    registerScalaFunction("log10", TFloat64(), TFloat64())(mathPackageClass, "log10")
    registerScalaFunction("sqrt", TFloat64(), TFloat64())(mathPackageClass, "sqrt")
    registerScalaFunction("log", TFloat64(), TFloat64())(mathPackageClass, "log")
    registerScalaFunction("log", TFloat64(), TFloat64(), TFloat64())(thisClass, "log")
    registerScalaFunction("gamma", TFloat64(), TFloat64())(thisClass, "gamma")

    registerScalaFunction("binomTest", TInt32(), TInt32(), TFloat64(), TInt32(), TFloat64())(statsPackageClass, "binomTest")

    registerScalaFunction("dbeta", TFloat64(), TFloat64(), TFloat64(), TFloat64())(statsPackageClass, "dbeta")

    registerScalaFunction("pnorm", TFloat64(), TFloat64())(statsPackageClass, "pnorm")
    registerScalaFunction("qnorm", TFloat64(), TFloat64())(statsPackageClass, "qnorm")

    registerScalaFunction("dpois", TFloat64(), TFloat64(), TFloat64())(statsPackageClass, "dpois")
    registerScalaFunction("dpois", TFloat64(), TFloat64(), TBoolean(), TFloat64())(statsPackageClass, "dpois")

    registerScalaFunction("ppois", TFloat64(), TFloat64(), TFloat64())(statsPackageClass, "ppois")
    registerScalaFunction("ppois", TFloat64(), TFloat64(), TBoolean(), TBoolean(), TFloat64())(statsPackageClass, "ppois")

    registerScalaFunction("qpois", TFloat64(), TFloat64(), TInt32())(statsPackageClass, "qpois")
    registerScalaFunction("qpois", TFloat64(), TFloat64(), TBoolean(), TBoolean(), TInt32())(statsPackageClass, "qpois")

    registerScalaFunction("pchisqtail", TFloat64(), TFloat64(), TFloat64())(statsPackageClass, "chiSquaredTail")
    registerScalaFunction("qchisqtail", TFloat64(), TFloat64(), TFloat64())(statsPackageClass, "inverseChiSquaredTail")

    registerScalaFunction("floor", TFloat32(), TFloat32())(thisClass, "floor")
    registerScalaFunction("floor", TFloat64(), TFloat64())(thisClass, "floor")

    registerScalaFunction("ceil", TFloat32(), TFloat32())(thisClass, "ceil")
    registerScalaFunction("ceil", TFloat64(), TFloat64())(thisClass, "ceil")

    registerJavaStaticFunction("%", TInt32(), TInt32(), TInt32())(jMathClass, "floorMod")
    registerJavaStaticFunction("%", TInt64(), TInt64(), TInt64())(jMathClass, "floorMod")
    registerScalaFunction("%", TFloat32(), TFloat32(), TFloat32())(thisClass, "mod")
    registerScalaFunction("%", TFloat64(), TFloat64(), TFloat64())(thisClass, "mod")

    registerJavaStaticFunction("isnan", TFloat32(), TBoolean())(jFloatClass, "isNaN")
    registerJavaStaticFunction("isnan", TFloat64(), TBoolean())(jDoubleClass, "isNaN")
  
    registerJavaStaticFunction("sign", TInt32(), TInt32())(jIntegerClass, "signum")
    registerScalaFunction("sign", TInt64(), TInt64())(mathPackageClass, "signum")
    registerJavaStaticFunction("sign", TFloat32(), TFloat32())(jMathClass, "signum")
    registerJavaStaticFunction("sign", TFloat64(), TFloat64())(jMathClass, "signum")

    registerWrappedScalaFunction("entropy", TString(), TFloat64())(thisClass, "irentropy")

    registerCode("fisher_exact_test", TInt32(), TInt32(), TInt32(), TInt32(), fetStruct){ case (mb, a, b, c, d) =>
      val res = mb.newLocal[Array[Double]]
      val srvb = new StagedRegionValueBuilder(mb, fetStruct)
      Code(
        res := Code.invokeScalaObject[Int, Int, Int, Int, Array[Double]](statsPackageClass, "fisherExactTest", a, b, c, d),
        srvb.start(),
        srvb.addDouble(res(0)),
        srvb.advance(),
        srvb.addDouble(res(1)),
        srvb.advance(),
        srvb.addDouble(res(2)),
        srvb.advance(),
        srvb.addDouble(res(3)),
        srvb.advance(),
        srvb.offset
      )
    }
    
    registerCode("chi_squared_test", TInt32(), TInt32(), TInt32(), TInt32(), chisqStruct){ case (mb, a, b, c, d) =>
      val res = mb.newLocal[Array[Double]]
      val srvb = new StagedRegionValueBuilder(mb, chisqStruct)
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

    registerCode("contingency_table_test", TInt32(), TInt32(), TInt32(), TInt32(), TInt32(), chisqStruct){ case (mb, a, b, c, d, min_cell_count) =>
      val res = mb.newLocal[Array[Double]]
      val srvb = new StagedRegionValueBuilder(mb, chisqStruct)
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

    registerCode("hardy_weinberg_test", TInt32(), TInt32(), TInt32(), hweStruct){ case (mb, nHomRef, nHet, nHomVar) =>
      val res = mb.newLocal[Array[Double]]
      val srvb = new StagedRegionValueBuilder(mb, hweStruct)
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

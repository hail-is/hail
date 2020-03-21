package is.hail.expr.ir.functions

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s.Code
import is.hail.expr.types._
import is.hail.expr.types.physical.{PArray, PFloat64}
import is.hail.expr.types.virtual.{TArray, TBoolean, TFloat64, TInt32}
import is.hail.utils._
import net.sourceforge.jdistlib.rng.MersenneTwister
import net.sourceforge.jdistlib.{Beta, Gamma, Poisson}

class IRRandomness(seed: Long) {

  // org.apache.commons has no way to statically sample from distributions without creating objects :(
  private[this] val random = new MersenneTwister()
  private[this] val poisState = Poisson.create_random_state()

  // FIXME: these are just combined with some large primes, so probably should be fixed up
  private[this] def hash(pidx: Int): Long =
    seed ^ java.lang.Math.floorMod(pidx * 11399L, 2147483647L)

  def reset(partitionIdx: Int) {
    val combinedSeed = hash(partitionIdx)
    random.setSeed(combinedSeed)
  }

  def runif(min: Double, max: Double): Double = min + (max - min) * random.nextDouble()

  def rcoin(p: Double): Boolean = random.nextDouble() < p

  def rpois(lambda: Double): Double = Poisson.random(lambda, random, poisState)

  def rnorm(mean: Double, sd: Double): Double = mean + sd * random.nextGaussian()

  def rbeta(a: Double, b: Double): Double = Beta.random(a, b, random)

  def rgamma(shape: Double, scale: Double): Double = Gamma.random(shape, scale, random)

  def rcat(prob: Array[Double]): Int = {
    var i = 0
    var sum = 0.0
    while (i < prob.length) {
      sum += prob(i)
      i += 1
    }
    var draw = random.nextDouble() * sum
    i = 0
    while (draw > prob(i)) {
      draw -= prob(i)
      i += 1
    }
    i
  }
}

object RandomSeededFunctions extends RegistryFunctions {

  def registerAll() {
    registerSeeded("rand_unif", TFloat64, TFloat64, TFloat64, null) { case (r, rt, seed, (minT, min), (maxT, max)) =>
      r.mb.newRNG(seed).invoke[Double, Double, Double]("runif", min, max)
    }

    registerSeeded("rand_norm", TFloat64, TFloat64, TFloat64, null) { case (r, rt, seed, (meanT, mean), (sdT, sd)) =>
      r.mb.newRNG(seed).invoke[Double, Double, Double]("rnorm", mean, sd)
    }

    registerSeeded("rand_bool", TFloat64, TBoolean, null) { case (r, rt, seed, (pT, p)) =>
      r.mb.newRNG(seed).invoke[Double, Boolean]("rcoin", p)
    }

    registerSeeded("rand_pois", TFloat64, TFloat64, null) { case (r, rt, seed, (lambdaT, lambda)) =>
      r.mb.newRNG(seed).invoke[Double, Double]("rpois", lambda)
    }

    registerSeeded("rand_pois", TInt32, TFloat64, TArray(TFloat64), null) { case (r, rt, seed, (nT, n), (lambdaT, lambda)) =>
      val length = r.mb.newLocal[Int]()
      val srvb = new StagedRegionValueBuilder(r, rt)
      Code(
        length := n,
        srvb.start(n),
        Code.whileLoop(srvb.arrayIdx < length,
          srvb.addDouble(r.mb.newRNG(seed).invoke[Double, Double]("rpois", lambda)),
          srvb.advance()
        ),
        srvb.offset)
    }

    registerSeeded("rand_beta", TFloat64, TFloat64, TFloat64, null) { case (r, rt, seed, (aT, a), (bT, b)) =>
      r.mb.newRNG(seed).invoke[Double, Double, Double]("rbeta", a, b)
    }

    registerSeeded("rand_beta", TFloat64, TFloat64, TFloat64, TFloat64, TFloat64, null) {
      case (r, rt, seed, (aT, a), (bT, b), (minT, min), (maxT, max)) =>
        val rng = r.mb.newRNG(seed)
        val la = r.mb.newLocal[Double]()
        val lb = r.mb.newLocal[Double]()
        val value = r.mb.newLocal[Double]()
        val lmin = r.mb.newLocal[Double]()
        val lmax = r.mb.newLocal[Double]()
        Code(
          la := a,
          lb := b,
          lmin := min,
          lmax := max,
          value := rng.invoke[Double, Double, Double]("rbeta", la, lb),
          Code.whileLoop(value < lmin || value > lmax,
            value := rng.invoke[Double, Double, Double]("rbeta", la, lb)),
          value)
    }

    registerSeeded("rand_gamma", TFloat64, TFloat64, TFloat64, null) { case (r, rt, seed, (aT, a), (scaleT, scale)) =>
      r.mb.newRNG(seed).invoke[Double, Double, Double]("rgamma", a, scale)
    }

    registerSeeded("rand_cat", TArray(TFloat64), TInt32, null) { case (r, rt, seed, (aT: PArray, a)) =>
      val array = r.mb.newLocal[Array[Double]]()
      val aoff = r.mb.newLocal[Long]()
      val length = r.mb.newLocal[Int]()
      val i = r.mb.newLocal[Int]()
      Code(
        aoff := a,
        i := 0,
        length := aT.loadLength(aoff),
        array := Code.newArray[Double](length),
        Code.whileLoop(i < length,
          array.load().update(i, Region.loadDouble(aT.elementOffset(aoff, length, i))),
          i += 1),
        r.mb.newRNG(seed).invoke[Array[Double], Int]("rcat", array))
    }
  }
}

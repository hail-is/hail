package is.hail.expr.ir.functions

import is.hail.annotations.StagedRegionValueBuilder
import is.hail.asm4s.Code
import is.hail.expr.types._
import net.sourceforge.jdistlib.Poisson
import net.sourceforge.jdistlib.rng.MersenneTwister

class IRRandomness(seed: Long) {

  private[this] val random = new MersenneTwister()
  private[this] var poisState = Poisson.create_random_state()

  // FIXME: these are just combined with some large primes, so probably should be fixed up
  private[this] def hash(pidx: Int): Long =
    seed ^ java.lang.Math.floorMod(pidx * 11399L, 2147483647L)

  def reset(partitionIdx: Int) {
    val combinedSeed = hash(partitionIdx)
    random.setSeed(combinedSeed)
    poisState = Poisson.create_random_state()
  }

  def runif(min: Double, max: Double): Double = min + (max - min) * random.nextDouble()

  def rcoin(p: Double): Boolean = random.nextDouble() < p

  def rpois(lambda: Double): Double = Poisson.random(lambda, random, poisState)

  def rnorm(mean: Double, sd: Double): Double = mean + sd * random.nextGaussian()
}

object RandomSeededFunctions extends RegistryFunctions {

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
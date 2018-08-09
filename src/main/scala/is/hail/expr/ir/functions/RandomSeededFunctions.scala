package is.hail.expr.ir.functions

import is.hail.annotations.StagedRegionValueBuilder
import is.hail.asm4s.Code
import is.hail.expr.types._
import org.apache.commons.math3.random.RandomDataGenerator

class IRRandomness(seed: Long) {

  protected[this] var pidx: Int = 0
  private[this] val random: RandomDataGenerator = new RandomDataGenerator()

  def setPartitionIndex(partitionIdx: Int): Unit =
    pidx = partitionIdx

  def reseed() {
    val combinedSeed = seed ^ java.lang.Math.floorMod(pidx * 11399, 17863)
    random.reSeed(combinedSeed)
  }

  def runif(min: Double, max: Double): Double = random.nextUniform(min, max, true)

  def rcoin(p: Double): Boolean = runif(0, 1) < p

  def rpois(lambda: Double): Double = random.nextPoisson(lambda)

  def rnorm(mean: Double, sd: Double): Double = random.nextGaussian(mean, sd)
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
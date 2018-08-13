package is.hail.expr.ir.functions

import breeze.linalg.{DenseVector, QuasiTensor}
import breeze.math.EnumeratedCoordinateField
import breeze.stats.distributions.{Beta, Dirichlet, Multinomial, RandBasis}
import is.hail.annotations.StagedRegionValueBuilder
import is.hail.asm4s.Code
import is.hail.expr.types._
import is.hail.utils._
import net.sourceforge.jdistlib
import org.apache.commons.math3.random.MersenneTwister

import scala.collection.mutable

class IRRandomness(seed: Long) {

  private[this] val random = new MersenneTwister()
  private[this] val randBasis = new RandBasis(random)
  private[this] val poisRandom = new jdistlib.rng.MersenneTwister()
  private[this] var poisState = jdistlib.Poisson.create_random_state()
  private[this] var multinomialDist: Multinomial[DenseVector[Double], Int] = _
  private[this] var dirichletDist: Dirichlet[DenseVector[Double], Int] = _

  // FIXME: these are just combined with some large primes, so probably should be fixed up
  private[this] def hash(pidx: Int): Long =
    seed ^ java.lang.Math.floorMod(pidx * 11399L, 2147483647L)

  def reset(partitionIdx: Int) {
    val combinedSeed = hash(partitionIdx)
    random.setSeed(combinedSeed)
    poisRandom.setSeed(combinedSeed)
    poisState = jdistlib.Poisson.create_random_state()
    multinomialDist = null
    dirichletDist = null
  }

  def runif(min: Double, max: Double): Double = min + (max - min) * random.nextDouble()

  def rcoin(p: Double): Boolean = random.nextDouble() < p

  def rpois(lambda: Double): Double = jdistlib.Poisson.random(lambda, poisRandom, poisState)

  def rnorm(mean: Double, sd: Double): Double = mean + sd * random.nextGaussian()

  def rbeta(a: Double, b: Double): Double = jdistlib.Beta.random(a, b, poisRandom)

  def multinomial(a: Array[Double]): Int = {
    val dv = DenseVector(a)
    if (multinomialDist == null || multinomialDist.params != dv) {
      val ev = implicitly[DenseVector[Double]=>QuasiTensor[Int, Double]]
      val sumImpl = implicitly[breeze.linalg.sum.Impl[DenseVector[Double], Double]]
      multinomialDist = Multinomial(dv)(ev, sumImpl, randBasis)
    }
    multinomialDist.draw()
  }

  def dirichlet(a: Array[Double]): Array[Double] = {
    val dv = DenseVector(a)
    if (dirichletDist == null || dirichletDist.params != dv) {
      val space = implicitly[EnumeratedCoordinateField[DenseVector[Double], Int, Double]]
      dirichletDist = Dirichlet(dv)(space, randBasis)
    }
    dirichletDist.draw().toArray
  }
}

object RandomSeededFunctions extends RegistryFunctions {

  def registerAll() {
    registerSeeded("runif_seeded", TFloat64(), TFloat64(), TFloat64()) { case (mb, seed, min, max) =>
      mb.newRNG(seed).invoke[Double, Double, Double]("runif", min, max)
    }

    registerSeeded("rnorm_seeded", TFloat64(), TFloat64(), TFloat64()) { case (mb, seed, mean, sd) =>
      mb.newRNG(seed).invoke[Double, Double, Double]("rnorm", mean, sd)
    }

    registerSeeded("pcoin_seeded", TFloat64(), TBoolean()) { case (mb, seed, p) =>
      mb.newRNG(seed).invoke[Double, Boolean]("rcoin", p)
    }

    registerSeeded("rpois_seeded", TFloat64(), TFloat64()) { case (mb, seed, lambda) =>
      mb.newRNG(seed).invoke[Double, Double]("rpois", lambda)
    }

    registerSeeded("rpois_seeded", TInt32(), TFloat64(), TArray(TFloat64())) { case (mb, seed, n, lambda) =>
      val length = mb.newLocal[Int]
      val srvb = new StagedRegionValueBuilder(mb, TArray(TFloat64()))
      Code(
        length := n,
        srvb.start(n),
        Code.whileLoop(srvb.arrayIdx < length,
          srvb.addDouble(mb.newRNG(seed).invoke[Double, Double]("rpois", lambda)),
          srvb.advance()
        ),
        srvb.offset)
    }

    registerSeeded("rbeta_seeded", TFloat64(), TFloat64(), TFloat64()) { case (mb, seed, a, b) =>
      mb.newRNG(seed).invoke[Double, Double, Double]("rbeta", a, b)
    }

    registerSeeded("rtruncatedbeta_seeded", TFloat64(), TFloat64(), TFloat64(), TFloat64(), TFloat64()) { case (mb, seed, a, b, min, max) =>
      val rng = mb.newRNG(seed)
      val value = mb.newLocal[Double]
      val lmin = mb.newLocal[Double]
      val lmax = mb.newLocal[Double]
      Code(
        lmin := min,
        lmax := max,
        value := rng.invoke[Double, Double, Double]("rbeta", a, b),
        Code.whileLoop(value < lmin || value > lmax,
          value := rng.invoke[Double, Double, Double]("rbeta", a, b)),
        value)
    }

    registerSeeded("rmultinomial_seeded", TArray(TFloat64()), TInt32()) { case (mb, seed, a) =>
      val tarray = TArray(TFloat64())
      val array = mb.newLocal[Array[Double]]
      val aoff = mb.newLocal[Long]
      val length = mb.newLocal[Int]
      val i = mb.newLocal[Int]
      Code(
        aoff := a,
        i := 0,
        length := tarray.loadLength(getRegion(mb), aoff),
        array := Code.newArray[Double](length),
        Code.whileLoop(i < length,
          array.load().update(i, getRegion(mb).loadDouble(tarray.elementOffset(aoff, length, i))),
          i += 1),
        mb.newRNG(seed).invoke[Array[Double], Int]("multinomial", array))
    }

    registerSeeded("rdirichlet_seeded", TArray(TFloat64()), TArray(TFloat64())) { case (mb, seed, a) =>
      val tarray = TArray(TFloat64())
      val array = mb.newLocal[Array[Double]]
      val aout = mb.newLocal[Array[Double]]
      val aoff = mb.newLocal[Long]
      val length = mb.newLocal[Int]
      val i = mb.newLocal[Int]
      val srvb = new StagedRegionValueBuilder(mb, tarray)
      Code(
        aoff := a,
        i := 0,
        length := tarray.loadLength(getRegion(mb), aoff),
        array := Code.newArray[Double](length),
        Code.whileLoop(i < length,
          array.load().update(i, getRegion(mb).loadDouble(tarray.elementOffset(aoff, length, i))),
          i += 1),
        aout := mb.newRNG(seed).invoke[Array[Double], Array[Double]]("dirichlet", array),
        length := aout.load().length(),
        srvb.start(length),
        Code.whileLoop(srvb.arrayIdx < length,
          srvb.addDouble(aout.load().apply(srvb.arrayIdx)),
          srvb.advance()),
        srvb.end())
    }
  }
}
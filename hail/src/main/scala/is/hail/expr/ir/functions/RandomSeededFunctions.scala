package is.hail.expr.ir.functions

import is.hail.asm4s._
import is.hail.expr.ir.IEmitCode
import is.hail.types.physical.stypes._
import is.hail.types.physical.stypes.concrete.SIndexablePointer
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives._
import is.hail.types.physical.{PBoolean, PCanonicalArray, PFloat64, PInt32, PType}
import is.hail.types.virtual._
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
    registerSeeded2("rand_unif", TFloat64, TFloat64, TFloat64, {
      case (_: Type, _: SType, _: SType) => SFloat64
    }) { case (cb, r, rt, seed, min, max) =>
      primitive(cb.emb.newRNG(seed).invoke[Double, Double, Double]("runif", min.asDouble.doubleCode(cb), max.asDouble.doubleCode(cb)))
    }

    registerSeeded2("rand_norm", TFloat64, TFloat64, TFloat64, {
      case (_: Type, _: SType, _: SType) => SFloat64
    }) { case (cb, r, rt, seed, mean, sd) =>
      primitive(cb.emb.newRNG(seed).invoke[Double, Double, Double]("rnorm", mean.asDouble.doubleCode(cb), sd.asDouble.doubleCode(cb)))
    }

    registerSeeded1("rand_bool", TFloat64, TBoolean, (_: Type, _: SType) => SBoolean) { case (cb, r, rt, seed, p) =>
      primitive(cb.emb.newRNG(seed).invoke[Double, Boolean]("rcoin", p.asDouble.doubleCode(cb)))
    }

    registerSeeded1("rand_pois", TFloat64, TFloat64, (_: Type, _: SType) => SFloat64) { case (cb, r, rt, seed, lambda) =>
      primitive(cb.emb.newRNG(seed).invoke[Double, Double]("rpois", lambda.asDouble.doubleCode(cb)))
    }

    registerSeeded2("rand_pois", TInt32, TFloat64, TArray(TFloat64), {
      case (_: Type, _: SType, _: SType) => PCanonicalArray(PFloat64(true)).sType
    }) { case (cb, r, SIndexablePointer(rt: PCanonicalArray), seed, n, lambdaCode) =>
      val len = cb.newLocal[Int]("rand_pos_len", n.asInt.intCode(cb))
      val lambda = cb.newLocal[Double]("rand_pois_lambda", lambdaCode.asDouble.doubleCode(cb))

      rt.constructFromElements(cb, r, len, deepCopy = false) { case (cb, _) =>
        IEmitCode.present(cb, primitive(cb.emb.newRNG(seed).invoke[Double, Double]("rpois", lambda)))
      }
    }

    registerSeeded2("rand_beta", TFloat64, TFloat64, TFloat64, {
      case (_: Type, _: SType, _: SType) => SFloat64
    }) { case (cb, r, rt, seed, a, b) =>
      primitive(
        cb.emb.newRNG(seed).invoke[Double, Double, Double]("rbeta",
          a.asDouble.doubleCode(cb), b.asDouble.doubleCode(cb)))
    }

    registerSeeded4("rand_beta", TFloat64, TFloat64, TFloat64, TFloat64, TFloat64, {
      case (_: Type, _: SType, _: SType, _: SType, _: SType) => SFloat64
    }) {
      case (cb, r, rt, seed, a, b, min, max) =>
        val rng = cb.emb.newRNG(seed)
        val la = cb.newLocal[Double]("la", a.asDouble.doubleCode(cb))
        val lb = cb.newLocal[Double]("lb", b.asDouble.doubleCode(cb))
        val lmin = cb.newLocal[Double]("lmin", min.asDouble.doubleCode(cb))
        val lmax = cb.newLocal[Double]("lmax", max.asDouble.doubleCode(cb))
        val value = cb.newLocal[Double]("value", rng.invoke[Double, Double, Double]("rbeta", la, lb))
        cb.whileLoop(value < lmin || value > lmax, {
          cb.assign(value, rng.invoke[Double, Double, Double]("rbeta", la, lb))
        })
        primitive(value)
    }

    registerSeeded2("rand_gamma", TFloat64, TFloat64, TFloat64, {
      case (_: Type, _: SType, _: SType) => SFloat64
    }) { case (cb, r, rt, seed, a, scale) =>
      primitive(
        cb.emb.newRNG(seed).invoke[Double, Double, Double]("rgamma", a.asDouble.doubleCode(cb), scale.asDouble.doubleCode(cb))
      )
    }

    registerSeeded1("rand_cat", TArray(TFloat64), TInt32, (_: Type, _: SType) => SInt32) { case (cb, r, rt, seed, aCode) =>
      val weights = aCode.asIndexable.memoize(cb, "rand_cat_weights")
      val len = weights.loadLength()

      val a = cb.newLocal[Array[Double]]("rand_cat_a", Code.newArray[Double](len))

      val i = cb.newLocal[Int]("rand_cat_i", 0)
      cb.whileLoop(i < len, {
        weights.loadElement(cb, i).consume(cb,
          cb._fatal("rand_cat requires all elements of input array to be present"),
          sc => cb += a.update(i, sc.asDouble.doubleCode(cb))
        )
        cb.assign(i, i + 1)
      })
      primitive(cb.emb.newRNG(seed).invoke[Array[Double], Int]("rcat", a))
    }
  }
}

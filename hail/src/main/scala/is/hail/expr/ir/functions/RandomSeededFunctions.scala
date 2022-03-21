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
import net.sourceforge.jdistlib.{Beta, Gamma, HyperGeometric, Poisson}

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

  def rhyper(numSuccessStates: Double, numFailureStates: Double, numToDraw: Double): Double = HyperGeometric.random(numSuccessStates, numFailureStates, numToDraw, random)

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
      primitive(cb.memoize(cb.emb.newRNG(seed).invoke[Double, Double, Double]("runif", min.asDouble.value, max.asDouble.value)))
    }

    registerSeeded2("rand_norm", TFloat64, TFloat64, TFloat64, {
      case (_: Type, _: SType, _: SType) => SFloat64
    }) { case (cb, r, rt, seed, mean, sd) =>
      primitive(cb.memoize(cb.emb.newRNG(seed).invoke[Double, Double, Double]("rnorm", mean.asDouble.value, sd.asDouble.value)))
    }

    registerSeeded1("rand_bool", TFloat64, TBoolean, (_: Type, _: SType) => SBoolean) { case (cb, r, rt, seed, p) =>
      primitive(cb.memoize(cb.emb.newRNG(seed).invoke[Double, Boolean]("rcoin", p.asDouble.value)))
    }

    registerSeeded1("rand_pois", TFloat64, TFloat64, (_: Type, _: SType) => SFloat64) { case (cb, r, rt, seed, lambda) =>
      primitive(cb.memoize(cb.emb.newRNG(seed).invoke[Double, Double]("rpois", lambda.asDouble.value)))
    }

    registerSeeded2("rand_pois", TInt32, TFloat64, TArray(TFloat64), {
      case (_: Type, _: SType, _: SType) => PCanonicalArray(PFloat64(true)).sType
    }) { case (cb, r, SIndexablePointer(rt: PCanonicalArray), seed, n, lambdaCode) =>
      val len = n.asInt.value
      val lambda = lambdaCode.asDouble.value

      rt.constructFromElements(cb, r, len, deepCopy = false) { case (cb, _) =>
        IEmitCode.present(cb, primitive(cb.memoize(cb.emb.newRNG(seed).invoke[Double, Double]("rpois", lambda))))
      }
    }

    registerSeeded2("rand_beta", TFloat64, TFloat64, TFloat64, {
      case (_: Type, _: SType, _: SType) => SFloat64
    }) { case (cb, r, rt, seed, a, b) =>
      primitive(cb.memoize(
        cb.emb.newRNG(seed).invoke[Double, Double, Double]("rbeta",
          a.asDouble.value, b.asDouble.value)))
    }

    registerSeeded4("rand_beta", TFloat64, TFloat64, TFloat64, TFloat64, TFloat64, {
      case (_: Type, _: SType, _: SType, _: SType, _: SType) => SFloat64
    }) {
      case (cb, r, rt, seed, a, b, min, max) =>
        val rng = cb.emb.newRNG(seed)
        val la = a.asDouble.value
        val lb = b.asDouble.value
        val lmin = min.asDouble.value
        val lmax = max.asDouble.value
        val value = cb.newLocal[Double]("value", rng.invoke[Double, Double, Double]("rbeta", la, lb))
        cb.whileLoop(value < lmin || value > lmax, {
          cb.assign(value, rng.invoke[Double, Double, Double]("rbeta", la, lb))
        })
        primitive(value)
    }

    registerSeeded2("rand_gamma", TFloat64, TFloat64, TFloat64, {
      case (_: Type, _: SType, _: SType) => SFloat64
    }) { case (cb, r, rt, seed, a, scale) =>
      primitive(cb.memoize(
        cb.emb.newRNG(seed).invoke[Double, Double, Double]("rgamma", a.asDouble.value, scale.asDouble.value)
      ))
    }

    registerSeeded1("rand_cat", TArray(TFloat64), TInt32, (_: Type, _: SType) => SInt32) { case (cb, r, rt, seed, weights: SIndexableValue) =>
      val len = weights.loadLength()

      val a = cb.newLocal[Array[Double]]("rand_cat_a", Code.newArray[Double](len))

      val i = cb.newLocal[Int]("rand_cat_i", 0)
      cb.whileLoop(i < len, {
        weights.loadElement(cb, i).consume(cb,
          cb._fatal("rand_cat requires all elements of input array to be present"),
          sc => cb += a.update(i, sc.asDouble.value)
        )
        cb.assign(i, i + 1)
      })
      primitive(cb.memoize(cb.emb.newRNG(seed).invoke[Array[Double], Int]("rcat", a)))
    }

    registerSeeded2("shuffle_compute_num_samples_per_partition", TInt32, TArray(TInt32), TArray(TInt32),
      (_, _, _) => SIndexablePointer(PCanonicalArray(PInt32(true), false))) { case (cb, r, rt, seed, initalNumSamplesToSelect: SInt32Value, partitionCounts: SIndexableValue) =>

      val totalNumberOfRecords = cb.newLocal[Int]("scnspp_total_number_of_records", 0)
      val resultSize: Value[Int] = partitionCounts.loadLength()
      val i = cb.newLocal[Int]("scnspp_index", 0)
      cb.forLoop(cb.assign(i, 0), i < resultSize, cb.assign(i, i + 1), {
        cb.assign(totalNumberOfRecords, totalNumberOfRecords + partitionCounts.loadElement(cb, i).get(cb).asInt32.value)
      })

      cb.ifx(initalNumSamplesToSelect.value > totalNumberOfRecords, cb._fatal("Requested selection of ", initalNumSamplesToSelect.value.toS,
        " samples from ", totalNumberOfRecords.toS, " records"))

      val successStatesRemaining = cb.newLocal[Int]("scnspp_success", initalNumSamplesToSelect.value)
      val failureStatesRemaining = cb.newLocal[Int]("scnspp_failure", totalNumberOfRecords - successStatesRemaining)

      val arrayRt = rt.asInstanceOf[SIndexablePointer]
      val (push, finish) = arrayRt.pType.asInstanceOf[PCanonicalArray].constructFromFunctions(cb, r, resultSize, false)

      cb.forLoop(cb.assign(i, 0), i < resultSize, cb.assign(i, i + 1), {
        val numSuccesses = cb.memoize(cb.emb.newRNG(seed).invoke[Double, Double, Double, Double]("rhyper",
          successStatesRemaining.toD, failureStatesRemaining.toD, partitionCounts.loadElement(cb, i).get(cb).asInt32.value.toD).toI)
        cb.assign(successStatesRemaining, successStatesRemaining - numSuccesses)
        cb.assign(failureStatesRemaining, failureStatesRemaining - (partitionCounts.loadElement(cb, i).get(cb).asInt32.value - numSuccesses))
        push(cb, IEmitCode.present(cb, new SInt32Value(numSuccesses)))
      })

      finish(cb)
    }

  }
}

package is.hail.expr.ir.functions

import is.hail.asm4s._
import is.hail.expr.Nat
import is.hail.expr.ir.{EmitCodeBuilder, IEmitCode}
import is.hail.types.physical.{PCanonicalArray, PCanonicalNDArray, PFloat64, PInt32}
import is.hail.types.physical.stypes._
import is.hail.types.physical.stypes.concrete.{SIndexablePointer, SNDArrayPointer, SRNGStateValue}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives._
import is.hail.types.virtual._
import is.hail.utils.FastSeq

import net.sourceforge.jdistlib.{Beta, Gamma, HyperGeometric, Poisson}
import net.sourceforge.jdistlib.rng.MersenneTwister

class IRRandomness(seed: Long) {

  /* org.apache.commons has no way to statically sample from distributions without creating objects
   * :( */
  private[this] val random = new MersenneTwister()
  private[this] val poisState = Poisson.create_random_state()

  // FIXME: these are just combined with some large primes, so probably should be fixed up
  private[this] def hash(pidx: Int): Long =
    seed ^ java.lang.Math.floorMod(pidx * 11399L, 2147483647L)

  def reset(partitionIdx: Int): Unit = {
    val combinedSeed = hash(partitionIdx)
    random.setSeed(combinedSeed)
  }

  def runif(min: Double, max: Double): Double = min + (max - min) * random.nextDouble()

  def rint32(n: Int): Int = random.nextInt(n)

  def rint64(): Long = random.nextLong()

  def rint64(n: Long): Long = random.nextLong(n)

  def rcoin(p: Double): Boolean = random.nextDouble() < p

  def rpois(lambda: Double): Double = Poisson.random(lambda, random, poisState)

  def rnorm(mean: Double, sd: Double): Double = mean + sd * random.nextGaussian()

  def rnorm(): Double = rnorm(0, 1)

  def rbeta(a: Double, b: Double): Double = Beta.random(a, b, random)

  def rgamma(shape: Double, scale: Double): Double = Gamma.random(shape, scale, random)

  def rhyper(numSuccessStates: Double, numFailureStates: Double, numToDraw: Double): Double =
    HyperGeometric.random(numSuccessStates, numFailureStates, numToDraw, random)

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

  def rand_unif(cb: EmitCodeBuilder, rand_longs: IndexedSeq[Value[Long]]): Code[Double] = {
    assert(rand_longs.length == 4)
    Code.invokeScalaObject4[Long, Long, Long, Long, Double](
      RandomSeededFunctions.getClass,
      "_rand_unif",
      rand_longs(0),
      rand_longs(1),
      rand_longs(2),
      rand_longs(3),
    )
  }

  // Equivalent to generating an infinite-precision real number in [0, 1),
  // represented as an infinitely long bitstream, and rounding down to the
  // nearest representable floating point number.
  // In contrast, the standard Java and jdistlib generators sample uniformly
  // from a sequence of equidistant floating point numbers in [0, 1), using
  // (nextLong() >>> 11).toDouble / (1L << 53)
  def _rand_unif(long0: Long, long1: Long, long2: Long, long3: Long): Double = {
    var bits = long3
    var exp = 1022
    if (bits == 0) {
      exp -= 64
      bits = long2
      if (bits == 0) {
        exp -= 64
        bits = long1
        if (bits == 0) {
          exp -= 64
          bits = long0
        }
      }
    }
    exp -= java.lang.Long.numberOfTrailingZeros(bits)
    java.lang.Double.longBitsToDouble(
      (exp.toLong << 52) | (long0 >>> 12)
    )
  }

  def registerAll(): Unit = {
    registerSCode3(
      "rand_unif",
      TRNGState,
      TFloat64,
      TFloat64,
      TFloat64,
      {
        case (_: Type, _: SType, _: SType, _: SType) => SFloat64
      },
    ) {
      case (_, cb, _, rngState: SRNGStateValue, min: SFloat64Value, max: SFloat64Value, _) =>
        primitive(cb.memoize(rand_unif(
          cb,
          rngState.rand(cb),
        ) * (max.value - min.value) + min.value))
    }

    registerSCode5(
      "rand_unif_nd",
      TRNGState,
      TInt64,
      TInt64,
      TFloat64,
      TFloat64,
      TNDArray(TFloat64, Nat(2)),
      {
        case (_: Type, _: SType, _: SType, _: SType, _: SType, _: SType) =>
          PCanonicalNDArray(PFloat64(true), 2, true).sType
      },
    ) {
      case (
            r,
            cb,
            rt: SNDArrayPointer,
            rngState: SRNGStateValue,
            nRows: SInt64Value,
            nCols: SInt64Value,
            min,
            max,
            _,
          ) =>
        val result = rt.pType.constructUninitialized(
          FastSeq(SizeValueDyn(nRows.value), SizeValueDyn(nCols.value)),
          cb,
          r.region,
        )
        val rng = cb.emb.getThreefryRNG()
        rngState.copyIntoEngine(cb, rng)
        result.coiterateMutate(cb, r.region) { _ =>
          primitive(cb.memoize(rng.invoke[Double, Double, Double](
            "runif",
            min.asDouble.value,
            max.asDouble.value,
          )))
        }
        result
    }

    registerSCode2(
      "rand_int32",
      TRNGState,
      TInt32,
      TInt32,
      {
        case (_: Type, _: SType, _: SType) => SInt32
      },
    ) { case (_, cb, _, rngState: SRNGStateValue, n: SInt32Value, _) =>
      val rng = cb.emb.getThreefryRNG()
      rngState.copyIntoEngine(cb, rng)
      primitive(cb.memoize(rng.invoke[Int, Int]("nextInt", n.value)))
    }

    registerSCode2(
      "rand_int64",
      TRNGState,
      TInt64,
      TInt64,
      {
        case (_: Type, _: SType, _: SType) => SInt64
      },
    ) { case (_, cb, _, rngState: SRNGStateValue, n: SInt64Value, _) =>
      val rng = cb.emb.getThreefryRNG()
      rngState.copyIntoEngine(cb, rng)
      primitive(cb.memoize(rng.invoke[Long, Long]("nextLong", n.value)))
    }

    registerSCode1(
      "rand_int64",
      TRNGState,
      TInt64,
      {
        case (_: Type, _: SType) => SInt64
      },
    ) { case (_, cb, _, rngState: SRNGStateValue, _) =>
      primitive(rngState.rand(cb)(0))
    }

    registerSCode5(
      "rand_norm_nd",
      TRNGState,
      TInt64,
      TInt64,
      TFloat64,
      TFloat64,
      TNDArray(TFloat64, Nat(2)),
      {
        case (_: Type, _: SType, _: SType, _: SType, _: SType, _: SType) =>
          PCanonicalNDArray(PFloat64(true), 2, true).sType
      },
    ) {
      case (
            r,
            cb,
            rt: SNDArrayPointer,
            rngState: SRNGStateValue,
            nRows: SInt64Value,
            nCols: SInt64Value,
            mean,
            sd,
            _,
          ) =>
        val result = rt.pType.constructUninitialized(
          FastSeq(SizeValueDyn(nRows.value), SizeValueDyn(nCols.value)),
          cb,
          r.region,
        )
        val rng = cb.emb.getThreefryRNG()
        rngState.copyIntoEngine(cb, rng)
        result.coiterateMutate(cb, r.region) { _ =>
          primitive(cb.memoize(rng.invoke[Double, Double, Double](
            "rnorm",
            mean.asDouble.value,
            sd.asDouble.value,
          )))
        }
        result
    }

    registerSCode3(
      "rand_norm",
      TRNGState,
      TFloat64,
      TFloat64,
      TFloat64,
      {
        case (_: Type, _: SType, _: SType, _: SType) => SFloat64
      },
    ) {
      case (_, cb, _, rngState: SRNGStateValue, mean: SFloat64Value, sd: SFloat64Value, _) =>
        val rng = cb.emb.getThreefryRNG()
        rngState.copyIntoEngine(cb, rng)
        primitive(cb.memoize(rng.invoke[Double, Double, Double]("rnorm", mean.value, sd.value)))
    }

    registerSCode2(
      "rand_bool",
      TRNGState,
      TFloat64,
      TBoolean,
      {
        case (_: Type, _: SType, _: SType) => SBoolean
      },
    ) { case (_, cb, _, rngState: SRNGStateValue, p: SFloat64Value, _) =>
      val u = rand_unif(cb, rngState.rand(cb))
      primitive(cb.memoize(u < p.value))
    }

    registerSCode2(
      "rand_pois",
      TRNGState,
      TFloat64,
      TFloat64,
      {
        case (_: Type, _: SType, _: SType) => SFloat64
      },
    ) { case (_, cb, _, rngState: SRNGStateValue, lambda: SFloat64Value, _) =>
      val rng = cb.emb.getThreefryRNG()
      rngState.copyIntoEngine(cb, rng)
      primitive(cb.memoize(rng.invoke[Double, Double]("rpois", lambda.value)))
    }

    registerSCode3(
      "rand_pois",
      TRNGState,
      TInt32,
      TFloat64,
      TArray(TFloat64),
      {
        case (_: Type, _: SType, _: SType, _: SType) => PCanonicalArray(PFloat64(true)).sType
      },
    ) {
      case (
            r,
            cb,
            SIndexablePointer(rt: PCanonicalArray),
            rngState: SRNGStateValue,
            n: SInt32Value,
            lambda: SFloat64Value,
            _,
          ) =>
        val rng = cb.emb.getThreefryRNG()
        rngState.copyIntoEngine(cb, rng)
        rt.constructFromElements(cb, r.region, n.value, deepCopy = false) { case (cb, _) =>
          IEmitCode.present(
            cb,
            primitive(cb.memoize(rng.invoke[Double, Double]("rpois", lambda.value))),
          )
        }
    }

    registerSCode3(
      "rand_beta",
      TRNGState,
      TFloat64,
      TFloat64,
      TFloat64,
      {
        case (_: Type, _: SType, _: SType, _: SType) => SFloat64
      },
    ) { case (_, cb, _, rngState: SRNGStateValue, a: SFloat64Value, b: SFloat64Value, _) =>
      val rng = cb.emb.getThreefryRNG()
      rngState.copyIntoEngine(cb, rng)
      primitive(cb.memoize(rng.invoke[Double, Double, Double]("rbeta", a.value, b.value)))
    }

    registerSCode5(
      "rand_beta",
      TRNGState,
      TFloat64,
      TFloat64,
      TFloat64,
      TFloat64,
      TFloat64,
      {
        case (_: Type, _: SType, _: SType, _: SType, _: SType, _: SType) => SFloat64
      },
    ) {
      case (
            _,
            cb,
            _,
            rngState: SRNGStateValue,
            a: SFloat64Value,
            b: SFloat64Value,
            min: SFloat64Value,
            max: SFloat64Value,
            _,
          ) =>
        val rng = cb.emb.getThreefryRNG()
        rngState.copyIntoEngine(cb, rng)
        val value = cb.newLocal[Double](
          "value",
          rng.invoke[Double, Double, Double]("rbeta", a.value, b.value),
        )
        cb.while_(
          value < min.value || value > max.value,
          cb.assign(value, rng.invoke[Double, Double, Double]("rbeta", a.value, b.value)),
        )
        primitive(value)
    }

    registerSCode3(
      "rand_gamma",
      TRNGState,
      TFloat64,
      TFloat64,
      TFloat64,
      {
        case (_: Type, _: SType, _: SType, _: SType) => SFloat64
      },
    ) {
      case (_, cb, _, rngState: SRNGStateValue, a: SFloat64Value, scale: SFloat64Value, _) =>
        val rng = cb.emb.getThreefryRNG()
        rngState.copyIntoEngine(cb, rng)
        primitive(cb.memoize(rng.invoke[Double, Double, Double]("rgamma", a.value, scale.value)))
    }

    registerSCode2(
      "rand_cat",
      TRNGState,
      TArray(TFloat64),
      TInt32,
      {
        case (_: Type, _: SType, _: SType) => SInt32
      },
    ) { case (_, cb, _, rngState: SRNGStateValue, weights: SIndexableValue, _) =>
      val len = weights.loadLength()
      val i = cb.newLocal[Int]("i", 0)
      val s = cb.newLocal[Double]("sum", 0.0)
      cb.while_(
        i < len, {
          cb.assign(
            s,
            s + weights.loadElement(cb, i).get(
              cb,
              "rand_cat requires all elements of input array to be present",
            ).asFloat64.value,
          )
          cb.assign(i, i + 1)
        },
      )
      val r = cb.newLocal[Double]("r", rand_unif(cb, rngState.rand(cb)) * s)
      cb.assign(i, 0)
      val elt = cb.newLocal[Double]("elt")
      cb.loop { start =>
        cb.assign(
          elt,
          weights.loadElement(cb, i).get(
            cb,
            "rand_cat requires all elements of input array to be present",
          ).asFloat64.value,
        )
        cb.if_(
          r > elt && i < len, {
            cb.assign(r, r - elt)
            cb.assign(i, i + 1)
            cb.goto(start)
          },
        )
      }
      primitive(i)
    }

    registerSCode3(
      "shuffle_compute_num_samples_per_partition",
      TRNGState,
      TInt32,
      TArray(TInt32),
      TArray(TInt32),
      (_, _, _, _) => SIndexablePointer(PCanonicalArray(PInt32(true), false)),
    ) {
      case (
            r,
            cb,
            rt,
            rngState: SRNGStateValue,
            initalNumSamplesToSelect: SInt32Value,
            partitionCounts: SIndexableValue,
            _,
          ) =>
        val rng = cb.emb.getThreefryRNG()
        rngState.copyIntoEngine(cb, rng)

        val totalNumberOfRecords = cb.newLocal[Int]("scnspp_total_number_of_records", 0)
        val resultSize: Value[Int] = partitionCounts.loadLength()
        val i = cb.newLocal[Int]("scnspp_index", 0)
        cb.for_(
          cb.assign(i, 0),
          i < resultSize,
          cb.assign(i, i + 1),
          cb.assign(
            totalNumberOfRecords,
            totalNumberOfRecords + partitionCounts.loadElement(cb, i).get(cb).asInt32.value,
          ),
        )

        cb.if_(
          initalNumSamplesToSelect.value > totalNumberOfRecords,
          cb._fatal(
            "Requested selection of ",
            initalNumSamplesToSelect.value.toS,
            " samples from ",
            totalNumberOfRecords.toS,
            " records",
          ),
        )

        val successStatesRemaining =
          cb.newLocal[Int]("scnspp_success", initalNumSamplesToSelect.value)
        val failureStatesRemaining =
          cb.newLocal[Int]("scnspp_failure", totalNumberOfRecords - successStatesRemaining)

        val arrayRt = rt.asInstanceOf[SIndexablePointer]
        val (push, finish) = arrayRt.pType.asInstanceOf[PCanonicalArray].constructFromFunctions(
          cb,
          r.region,
          resultSize,
          false,
        )

        cb.for_(
          cb.assign(i, 0),
          i < resultSize,
          cb.assign(i, i + 1), {
            val numSuccesses = cb.memoize(rng.invoke[Double, Double, Double, Double](
              "rhyper",
              successStatesRemaining.toD,
              failureStatesRemaining.toD,
              partitionCounts.loadElement(cb, i).get(cb).asInt32.value.toD,
            ).toI)
            cb.assign(successStatesRemaining, successStatesRemaining - numSuccesses)
            cb.assign(
              failureStatesRemaining,
              failureStatesRemaining - (partitionCounts.loadElement(cb, i).get(
                cb
              ).asInt32.value - numSuccesses),
            )
            push(cb, IEmitCode.present(cb, new SInt32Value(numSuccesses)))
          },
        )

        finish(cb)
    }

  }
}

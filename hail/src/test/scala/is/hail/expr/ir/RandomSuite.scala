package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.asm4s._
import is.hail.types.physical.stypes.concrete.{SCanonicalRNGStateSettable, SCanonicalRNGStateValue, SRNGState, SRNGStateStaticSizeValue}
import is.hail.utils.FastIndexedSeq
import org.apache.commons.math3.distribution.ChiSquaredDistribution
import org.testng.annotations.Test

class RandomSuite extends HailSuite {
  // from skein_golden_kat_short_internals.txt in the skein source
  val threefryTestCases = FastIndexedSeq(
    (
      Array(0x0L, 0x0L, 0x0L, 0x0L),
      Array(0x0L, 0x0L),
      Array(0x0L, 0x0L, 0x0L, 0x0L),
      Array(0x09218EBDE6C85537L, 0x55941F5266D86105L, 0x4BD25E16282434DCL, 0xEE29EC846BD2E40BL)
    ), (
      Array(0x1716151413121110L, 0x1F1E1D1C1B1A1918L, 0x2726252423222120L, 0x2F2E2D2C2B2A2928L),
      Array(0x0706050403020100L, 0x0F0E0D0C0B0A0908L),
      Array(0xF8F9FAFBFCFDFEFFL, 0xF0F1F2F3F4F5F6F7L, 0xE8E9EAEBECEDEEEFL, 0xE0E1E2E3E4E5E6E7L),
      Array(0x008CF75D18C19DA0L, 0x1D7D14BE2266E7D8L, 0x5D09E0E985FE673BL, 0xB4A5480C6039B172L)
    ))

  @Test def testThreefry() {
    for {
      (key, tweak, input, expected) <- threefryTestCases
    } {
      val expandedKey = Threefry.expandKey(key)
      val tf = Threefry(key)

      var x = input.clone()
      tf(x, tweak)
      assert(x sameElements expected)

      x = input.clone()
      Threefry.encryptUnrolled(
        expandedKey(0), expandedKey(1), expandedKey(2), expandedKey(3), expandedKey(4),
        tweak(0), tweak(1), x)
      assert(x sameElements expected)

      x = input.clone()
      Threefry.encrypt(expandedKey, tweak, x)
      assert(x sameElements expected)
    }
  }

  def pmacStagedStaticSize(staticID: Long, size: Int): AsmFunction1[Array[Long], Array[Long]] = {
    val f = EmitFunctionBuilder[Array[Long], Array[Long]](ctx, "pmacStaticSize")
    f.emb.emitWithBuilder { cb =>
      val message = f.mb.getArg[Array[Long]](1)
      var state = SRNGStateStaticSizeValue(cb)
      for (i <- 0 until size) {
        state = state.splitDyn(cb, cb.memoize(message(i)))
      }
      state = state.splitStatic(cb, staticID)

      val result = state.rand(cb)
      val resArray = cb.memoize(Code.newArray[Long](4))
      cb.append(resArray(0) = result(0))
      cb.append(resArray(1) = result(1))
      cb.append(resArray(2) = result(2))
      cb.append(resArray(3) = result(3))

      resArray
    }
    f.result(ctx)(new HailClassLoader(getClass.getClassLoader))
  }

  def pmacEngineStagedStaticSize(staticID: Long, size: Int): AsmFunction1[Array[Long], ThreefryRandomEngine] = {
    val f = EmitFunctionBuilder[Array[Long], ThreefryRandomEngine](ctx, "pmacStaticSize")
    f.emb.emitWithBuilder { cb =>
      val message = f.mb.getArg[Array[Long]](1)
      var state = SRNGStateStaticSizeValue(cb)
      for (i <- 0 until size) {
        state = state.splitDyn(cb, cb.memoize(message(i)))
      }
      state = state.splitStatic(cb, staticID)

      val engine = cb.memoize(Code.invokeScalaObject0[ThreefryRandomEngine](
        ThreefryRandomEngine.getClass, "apply"))
      state.copyIntoEngine(cb, engine)
      engine
    }
    f.result(ctx)(new HailClassLoader(getClass.getClassLoader))
  }

  def pmacStagedDynSize(staticID: Long): AsmFunction1[Array[Long], Array[Long]] = {
    val f = EmitFunctionBuilder[Array[Long], Array[Long]](ctx, "pmacDynSize")
    f.emb.emitWithBuilder { cb =>
      val message = f.mb.getArg[Array[Long]](1)
      val state = cb.newSLocal(SRNGState(None), "state").asInstanceOf[SCanonicalRNGStateSettable]
      cb.assign(state, SCanonicalRNGStateValue(cb))
      val i = cb.newLocal[Int]("i", 0)
      val len = cb.memoize(message.length())
      cb.forLoop({}, i < len, cb.assign(i, i + 1), {
        cb.assign(state, state.splitDyn(cb, cb.memoize(message(i))))
      })
      cb.assign(state, state.splitStatic(cb, staticID))

      val result = state.rand(cb)
      val resArray = cb.memoize(Code.newArray[Long](4))
      cb.append(resArray(0) = result(0))
      cb.append(resArray(1) = result(1))
      cb.append(resArray(2) = result(2))
      cb.append(resArray(3) = result(3))

      resArray
    }
    f.result(ctx)(new HailClassLoader(getClass.getClassLoader))
  }

  def pmacEngineStagedDynSize(staticID: Long): AsmFunction1[Array[Long], ThreefryRandomEngine] = {
    val f = EmitFunctionBuilder[Array[Long], ThreefryRandomEngine](ctx, "pmacDynSize")
    f.emb.emitWithBuilder { cb =>
      val message = f.mb.getArg[Array[Long]](1)
      val state = cb.newSLocal(SRNGState(None), "state").asInstanceOf[SCanonicalRNGStateSettable]
      cb.assign(state, SCanonicalRNGStateValue(cb))
      val i = cb.newLocal[Int]("i", 0)
      val len = cb.memoize(message.length())
      cb.forLoop({}, i < len, cb.assign(i, i + 1), {
        cb.assign(state, state.splitDyn(cb, cb.memoize(message(i))))
      })
      cb.assign(state, state.splitStatic(cb, staticID))

      val engine = cb.memoize(Code.invokeScalaObject0[ThreefryRandomEngine](
        ThreefryRandomEngine.getClass, "apply"))
      state.copyIntoEngine(cb, engine)
      engine
    }
    f.result(ctx)(new HailClassLoader(getClass.getClassLoader))
  }

  val pmacTestCases = FastIndexedSeq(
    (Array[Long](), 0L),
    (Array[Long](100, 101), 10L),
    (Array[Long](100, 101, 102, 103), 20L),
    (Array[Long](100, 101, 102, 103, 104), 30L)
  )

  @Test def testPMAC() {
    for {
      (message, staticID) <- pmacTestCases
    } {
      val res1 = Threefry.pmac(ctx.rngNonce, staticID, message)
      val res2 = pmacStagedStaticSize(staticID, message.length)(message)
      val res3 = pmacStagedDynSize(staticID)(message)
      assert(res1 sameElements res2)
      assert(res1 sameElements res3)
    }
  }

  @Test def testPMACHash() {
    for {
      (message, _) <- pmacTestCases
    } {
      val res1 = Threefry.pmac(message)
      val res2 = new PMacHash().extend(message).hash
      val n = message.length
      val res3 = new PMacHash().extend(message.slice(0, n / 2)).extend(message.slice(n / 2, n)).hash
      assert(res1 sameElements res2)
      assert(res1 sameElements res3)
    }
  }

  @Test def testRandomEngine() {
    for {
      (message, staticID) <- pmacTestCases
    } {
      val (hash, finalTweak) = Threefry.pmacHash(ctx.rngNonce, staticID, message)
      val engine1 = pmacEngineStagedStaticSize(staticID, message.length)(message)
      val engine2 = pmacEngineStagedDynSize(staticID)(message)

      var expected = hash.clone()
      Threefry.encrypt(Threefry.defaultKey, Array(finalTweak, 0L), expected)
      assert(Array.fill(4)(engine1.nextLong()) sameElements expected)
      assert(Array.fill(4)(engine2.nextLong()) sameElements expected)

      expected = hash.clone()
      Threefry.encrypt(Threefry.defaultKey, Array(finalTweak, 1L), expected)
      assert(Array.fill(4)(engine1.nextLong()) sameElements expected)
      assert(Array.fill(4)(engine2.nextLong()) sameElements expected)

      expected = hash.clone()
      Threefry.encrypt(Threefry.defaultKey, Array(finalTweak, 2L), expected)
      assert(Array.fill(4)(engine1.nextLong()) sameElements expected)
      assert(Array.fill(4)(engine2.nextLong()) sameElements expected)
    }
  }

  def runChiSquareTest(samples: Int, buckets: Int)(sample: => Int) {
    val chiSquareDist = new ChiSquaredDistribution(buckets - 1)
    val expected = samples.toDouble / buckets
    var numRuns = 0
    val passThreshold = 0.1
    val failThreshold = 1e-6
    var geometricMean = failThreshold

    while (geometricMean >= failThreshold && geometricMean < passThreshold) {
      val counts = Array.ofDim[Int](buckets)
      for (_ <- 0 until samples) counts(sample) += 1
      val chisquare = counts.map(observed => math.pow(observed - expected, 2) / expected).sum
      val pvalue = 1 - chiSquareDist.cumulativeProbability(chisquare)
      numRuns += 1
      geometricMean = math.pow(geometricMean, (numRuns - 1).toDouble / numRuns) * math.pow(pvalue, 1.0 / numRuns)
    }
    assert(geometricMean >= passThreshold, s"failed after $numRuns runs with pvalue $geometricMean")
    println(s"passed after $numRuns runs with pvalue $geometricMean")
  }

  @Test def testRandomInt() {
    val n = 1 << 25
    val k = 1 << 15
    val rand = ThreefryRandomEngine.randState()
    runChiSquareTest(n, k) {
      rand.nextInt() & (k - 1)
    }
  }

  @Test def testBoundedUniformInt() {
    var n = 1 << 25
    var k = 1 << 15
    val rand = ThreefryRandomEngine.randState()
    runChiSquareTest(n, k) {
      rand.nextInt(k)
    }

    n = 30000000
    k = math.pow(n, 3.0/5).toInt
    runChiSquareTest(n, k) {
      rand.nextInt(k)
    }
  }

  @Test def testBoundedUniformLong() {
    var n = 1 << 25
    var k = 1 << 15
    val rand = ThreefryRandomEngine.randState()
    runChiSquareTest(n, k) {
      rand.nextLong(k).toInt
    }

    n = 30000000
    k = math.pow(n, 3.0/5).toInt
    runChiSquareTest(n, k) {
      rand.nextLong(k).toInt
    }
  }

  @Test def testUniformDouble() {
    val n = 1 << 25
    val k = 1 << 15
    val rand = ThreefryRandomEngine.randState()
    runChiSquareTest(n, k) {
      val r = rand.nextDouble()
      assert(r >= 0.0 && r < 1.0, r)
      (r * k).toInt
    }
  }

  @Test def testUniformFloat() {
    val n = 1 << 25
    val k = 1 << 15
    val rand = ThreefryRandomEngine.randState()
    runChiSquareTest(n, k) {
      val r = rand.nextFloat()
      assert(r >= 0.0 && r < 1.0, r)
      (r * k).toInt
    }
  }
}

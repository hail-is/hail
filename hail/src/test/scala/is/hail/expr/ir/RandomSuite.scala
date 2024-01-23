package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.asm4s._
import is.hail.types.physical.stypes.concrete.{
  SCanonicalRNGStateSettable, SCanonicalRNGStateValue, SRNGState, SRNGStateStaticSizeValue,
}
import is.hail.utils.FastSeq

import org.apache.commons.math3.distribution.ChiSquaredDistribution
import org.testng.annotations.Test

class RandomSuite extends HailSuite {
  // from skein_golden_kat_short_internals.txt in the skein source
  val threefryTestCases = FastSeq(
    (
      Array(0x0L, 0x0L, 0x0L, 0x0L),
      Array(0x0L, 0x0L),
      Array(0x0L, 0x0L, 0x0L, 0x0L),
      Array(0x09218ebde6c85537L, 0x55941f5266d86105L, 0x4bd25e16282434dcL, 0xee29ec846bd2e40bL),
    ),
    (
      Array(0x1716151413121110L, 0x1f1e1d1c1b1a1918L, 0x2726252423222120L, 0x2f2e2d2c2b2a2928L),
      Array(0x0706050403020100L, 0x0f0e0d0c0b0a0908L),
      Array(0xf8f9fafbfcfdfeffL, 0xf0f1f2f3f4f5f6f7L, 0xe8e9eaebecedeeefL, 0xe0e1e2e3e4e5e6e7L),
      Array(0x008cf75d18c19da0L, 0x1d7d14be2266e7d8L, 0x5d09e0e985fe673bL, 0xb4a5480c6039b172L),
    ),
  )

  @Test def testThreefry(): Unit = {
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
        expandedKey(0),
        expandedKey(1),
        expandedKey(2),
        expandedKey(3),
        expandedKey(4),
        tweak(0),
        tweak(1),
        x,
      )
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
      for (i <- 0 until size)
        state = state.splitDyn(cb, cb.memoize(message(i)))
      state = state.splitStatic(cb, staticID)

      val result = state.rand(cb)
      val resArray = cb.memoize(Code.newArray[Long](4))
      cb.append(resArray(0) = result(0))
      cb.append(resArray(1) = result(1))
      cb.append(resArray(2) = result(2))
      cb.append(resArray(3) = result(3))

      resArray
    }
    f.result()(new HailClassLoader(getClass.getClassLoader))
  }

  def pmacEngineStagedStaticSize(staticID: Long, size: Int)
    : AsmFunction1[Array[Long], ThreefryRandomEngine] = {
    val f = EmitFunctionBuilder[Array[Long], ThreefryRandomEngine](ctx, "pmacStaticSize")
    f.emb.emitWithBuilder { cb =>
      val message = f.mb.getArg[Array[Long]](1)
      var state = SRNGStateStaticSizeValue(cb)
      for (i <- 0 until size)
        state = state.splitDyn(cb, cb.memoize(message(i)))
      state = state.splitStatic(cb, staticID)

      val engine = cb.memoize(Code.invokeScalaObject0[ThreefryRandomEngine](
        ThreefryRandomEngine.getClass,
        "apply",
      ))
      state.copyIntoEngine(cb, engine)
      engine
    }
    f.result()(new HailClassLoader(getClass.getClassLoader))
  }

  def pmacStagedDynSize(staticID: Long): AsmFunction1[Array[Long], Array[Long]] = {
    val f = EmitFunctionBuilder[Array[Long], Array[Long]](ctx, "pmacDynSize")
    f.emb.emitWithBuilder { cb =>
      val message = f.mb.getArg[Array[Long]](1)
      val state = cb.newSLocal(SRNGState(None), "state").asInstanceOf[SCanonicalRNGStateSettable]
      cb.assign(state, SCanonicalRNGStateValue(cb))
      val i = cb.newLocal[Int]("i", 0)
      val len = cb.memoize(message.length())
      cb.for_(
        {},
        i < len,
        cb.assign(i, i + 1),
        cb.assign(state, state.splitDyn(cb, cb.memoize(message(i)))),
      )
      cb.assign(state, state.splitStatic(cb, staticID))

      val result = state.rand(cb)
      val resArray = cb.memoize(Code.newArray[Long](4))
      cb.append(resArray(0) = result(0))
      cb.append(resArray(1) = result(1))
      cb.append(resArray(2) = result(2))
      cb.append(resArray(3) = result(3))

      resArray
    }
    f.result()(new HailClassLoader(getClass.getClassLoader))
  }

  def pmacEngineStagedDynSize(staticID: Long): AsmFunction1[Array[Long], ThreefryRandomEngine] = {
    val f = EmitFunctionBuilder[Array[Long], ThreefryRandomEngine](ctx, "pmacDynSize")
    f.emb.emitWithBuilder { cb =>
      val message = f.mb.getArg[Array[Long]](1)
      val state = cb.newSLocal(SRNGState(None), "state").asInstanceOf[SCanonicalRNGStateSettable]
      cb.assign(state, SCanonicalRNGStateValue(cb))
      val i = cb.newLocal[Int]("i", 0)
      val len = cb.memoize(message.length())
      cb.for_(
        {},
        i < len,
        cb.assign(i, i + 1),
        cb.assign(state, state.splitDyn(cb, cb.memoize(message(i)))),
      )
      cb.assign(state, state.splitStatic(cb, staticID))

      val engine = cb.memoize(Code.invokeScalaObject0[ThreefryRandomEngine](
        ThreefryRandomEngine.getClass,
        "apply",
      ))
      state.copyIntoEngine(cb, engine)
      engine
    }
    f.result()(new HailClassLoader(getClass.getClassLoader))
  }

  val pmacTestCases = FastSeq(
    (Array[Long](), 0L),
    (Array[Long](100, 101), 10L),
    (Array[Long](100, 101, 102, 103), 20L),
    (Array[Long](100, 101, 102, 103, 104), 30L),
  )

  @Test def testPMAC(): Unit = {
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

  @Test def testPMACHash(): Unit = {
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

  @Test def testRandomEngine(): Unit = {
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

  def runChiSquareTest(samples: Int, buckets: Int)(sample: => Int): Unit = {
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
      geometricMean =
        math.pow(geometricMean, (numRuns - 1).toDouble / numRuns) * math.pow(pvalue, 1.0 / numRuns)
    }
    assert(geometricMean >= passThreshold, s"failed after $numRuns runs with pvalue $geometricMean")
    println(s"passed after $numRuns runs with pvalue $geometricMean")
  }

  @Test def testRandomInt(): Unit = {
    val n = 1 << 25
    val k = 1 << 15
    val rand = ThreefryRandomEngine.randState()
    runChiSquareTest(n, k) {
      rand.nextInt() & (k - 1)
    }
  }

  @Test def testBoundedUniformInt(): Unit = {
    var n = 1 << 25
    var k = 1 << 15
    val rand = ThreefryRandomEngine.randState()
    runChiSquareTest(n, k) {
      rand.nextInt(k)
    }

    n = 30000000
    k = math.pow(n, 3.0 / 5).toInt
    runChiSquareTest(n, k) {
      rand.nextInt(k)
    }
  }

  @Test def testBoundedUniformLong(): Unit = {
    var n = 1 << 25
    var k = 1 << 15
    val rand = ThreefryRandomEngine.randState()
    runChiSquareTest(n, k) {
      rand.nextLong(k).toInt
    }

    n = 30000000
    k = math.pow(n, 3.0 / 5).toInt
    runChiSquareTest(n, k) {
      rand.nextLong(k).toInt
    }
  }

  @Test def testUniformDouble(): Unit = {
    val n = 1 << 25
    val k = 1 << 15
    val rand = ThreefryRandomEngine.randState()
    runChiSquareTest(n, k) {
      val r = rand.nextDouble()
      assert(r >= 0.0 && r < 1.0, r)
      (r * k).toInt
    }
  }

  @Test def testUniformFloat(): Unit = {
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

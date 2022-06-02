package is.hail.expr.ir

import is.hail.HailSuite
import org.apache.commons.math3.distribution.ChiSquaredDistribution
import org.testng.annotations.Test

class RandomSuite extends HailSuite {
  @Test def testThreefry() {
    val k = Array.fill[Long](4)(0)
    val tf = Threefry(k)
    val x = Array.fill[Long](4)(0)
    val expected = Array(
      0x09218EBDE6C85537L,
      0x55941F5266D86105L,
      0x4BD25E16282434DCL,
      0xEE29EC846BD2E40BL
    )
    tf(x, 0)
    assert(x sameElements expected)

    val rand = new ThreefryRandomEngine(k, Array.fill(4)(0L), 0, tweak = 0)
    val y = Array.fill(4)(rand.nextLong())
    assert(y sameElements expected)
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
    val rand = ThreefryRandomEngine()
    runChiSquareTest(n, k) {
      rand.nextInt() & (k - 1)
    }
  }

  @Test def testBoundedUniformInt() {
    var n = 1 << 25
    var k = 1 << 15
    val rand = ThreefryRandomEngine()
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
    val rand = ThreefryRandomEngine()
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
    val rand = ThreefryRandomEngine()
    runChiSquareTest(n, k) {
      val r = rand.nextDouble()
      assert(r >= 0.0 && r < 1.0, r)
      (r * k).toInt
    }
  }

  @Test def testUniformFloat() {
    val n = 1 << 25
    val k = 1 << 15
    val rand = ThreefryRandomEngine()
    runChiSquareTest(n, k) {
      val r = rand.nextFloat()
      assert(r >= 0.0 && r < 1.0, r)
      (r * k).toInt
    }
  }
}

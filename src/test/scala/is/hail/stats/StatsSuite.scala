package is.hail.stats

import breeze.linalg.DenseMatrix
import org.apache.commons.math3.distribution.{ChiSquaredDistribution, NormalDistribution}
import is.hail.utils._
import is.hail.variant.Variant
import is.hail.SparkSuite
import org.testng.annotations.Test

class StatsSuite extends SparkSuite {

  @Test def chiSquaredTailTest() {
    val chiSq1 = new ChiSquaredDistribution(1)
    assert(D_==(chiSquaredTail(1,1d), 1 - chiSq1.cumulativeProbability(1d)))
    assert(D_==(chiSquaredTail(1,5.52341d), 1 - chiSq1.cumulativeProbability(5.52341d)))

    val chiSq2 = new ChiSquaredDistribution(2)
    assert(D_==(chiSquaredTail(2, 1), 1 - chiSq2.cumulativeProbability(1)))
    assert(D_==(chiSquaredTail(2, 5.52341), 1 - chiSq2.cumulativeProbability(5.52341)))

    val chiSq5 = new ChiSquaredDistribution(5.2)
    assert(D_==(chiSquaredTail(5.2, 1), 1 - chiSq5.cumulativeProbability(1)))
    assert(D_==(chiSquaredTail(5.2, 5.52341), 1 - chiSq5.cumulativeProbability(5.52341)))

    assert(D_==(inverseChiSquaredTail(1.0, .1), chiSq1.inverseCumulativeProbability(1 - .1)))
    assert(D_==(inverseChiSquaredTail(1.0, .0001), chiSq1.inverseCumulativeProbability(1 - .0001)))

    val a = List(.0000000001, .5, .9999999999, 1.0)
    a.foreach(p => assert(D_==(chiSquaredTail(1.0, inverseChiSquaredTail(1.0, p)), p)))

    // compare with R
    assert(math.abs(chiSquaredTail(1, 400) - 5.507248e-89) < 1e-93)
    assert(D_==(inverseChiSquaredTail(1, 5.507248e-89), 400))
  }

  @Test def normalTest() {
    val normalDist = new NormalDistribution()
    assert(D_==(pnorm(1), normalDist.cumulativeProbability(1)))
    assert(math.abs(pnorm(-10) - normalDist.cumulativeProbability(-10)) < 1e-10)
    assert(D_==(qnorm(.6), normalDist.inverseCumulativeProbability(.6)))
    assert(D_==(qnorm(.0001), normalDist.inverseCumulativeProbability(.0001)))

    val a = List(0.0, .0000000001, .5, .9999999999, 1.0)
    assert(a.forall(p => D_==(qnorm(pnorm(qnorm(p))), qnorm(p))))

    // compare with R
    assert(math.abs(pnorm(-20) - 2.753624e-89) < 1e-93)
    assert(D_==(qnorm(2.753624e-89), -20))
  }

  @Test def vdsFromMatrixTest() {
    val G = DenseMatrix((0, 1), (2, -1), (0, 1))
    val vds = vdsFromMatrix(hc)(G)

    val G1 = DenseMatrix.zeros[Int](3, 2)
    vds.rdd.collect().foreach{ case (v, (va, gs)) => gs.zipWithIndex.foreach { case (g, i) => G1(i, v.start - 1) = g.gt.getOrElse(-1) } }

    assert(vds.sampleIds == IndexedSeq("0", "1", "2"))
    assert(vds.variants.collect().toSet == Set(Variant("1", 1, "A", "C"), Variant("1", 2, "A", "C")))

    for (i <- 0 to 2)
      for (j <- 0 to 1)
        assert(G(i, j) == G1(i, j))
  }
}

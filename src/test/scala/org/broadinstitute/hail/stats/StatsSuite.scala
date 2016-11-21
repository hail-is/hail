package org.broadinstitute.hail.stats

import breeze.linalg.DenseMatrix
import org.apache.commons.math3.distribution.{ChiSquaredDistribution, NormalDistribution}
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant.Variant
import org.broadinstitute.hail.SparkSuite
import org.testng.annotations.Test

class StatsSuite extends SparkSuite {

  @Test def chiSquaredTailTest() = {
    val chiSq1 = new ChiSquaredDistribution(1)
    assert(D_==(chiSquaredTail(1,1d), 1 - chiSq1.cumulativeProbability(1d)))
    assert(D_==(chiSquaredTail(1,5.52341d), 1 - chiSq1.cumulativeProbability(5.52341d)))

    val chiSq2 = new ChiSquaredDistribution(2)
    assert(D_==(chiSquaredTail(2, 1), 1 - chiSq2.cumulativeProbability(1)))
    assert(D_==(chiSquaredTail(2, 5.52341), 1 - chiSq2.cumulativeProbability(5.52341)))

    val chiSq5 = new ChiSquaredDistribution(5.2)
    assert(D_==(chiSquaredTail(5.2, 1), 1 - chiSq5.cumulativeProbability(1)))
    assert(D_==(chiSquaredTail(5.2, 5.52341), 1 - chiSq5.cumulativeProbability(5.52341)))

    assert(D_==(inverseChiSquaredTailOneDF(.1), chiSq1.inverseCumulativeProbability(1 - .1)))
    assert(D_==(inverseChiSquaredTailOneDF(.0001), chiSq1.inverseCumulativeProbability(1 - .0001)))

    val a = List(.0000000001, .5, .9999999999, 1.0)
    a.foreach(p => println(p, inverseChiSquaredTailOneDF(p)))
    a.foreach(p => assert(D_==(chiSquaredTail(1.0, inverseChiSquaredTailOneDF(p)), p)))
  }

  @Test def normTest() = {
    val normalDist = new NormalDistribution()
    assert(D_==(pnorm(1), normalDist.cumulativeProbability(1)))
    assert(D_==(pnorm(-10), normalDist.cumulativeProbability(-10)))
    assert(D_==(qnorm(.6), normalDist.inverseCumulativeProbability(.6)))
    assert(D_==(qnorm(.0001), normalDist.inverseCumulativeProbability(.0001)))

    val a = List(0.0, .0000000001, .5, .9999999999, 1.0)
    assert(a.forall(p => D_==(qnorm(pnorm(qnorm(p))), qnorm(p))))
  }

  @Test def vdsFromMatrixTest() {
    val G = DenseMatrix((0, 1), (2, -1), (0, 1))
    val vds = vdsFromMatrix(sc)(G)

    val G1 = DenseMatrix.zeros[Int](3, 2)
    vds.rdd.collect().foreach{ case (v, (va, gs)) => gs.zipWithIndex.foreach { case (g, i) => G1(i, v.start - 1) = g.gt.getOrElse(-1) } }

    assert(vds.sampleIds == IndexedSeq("0", "1", "2"))
    assert(vds.variants.collect().toSet == Set(Variant("1", 1, "A", "C"), Variant("1", 2, "A", "C")))

    for (i <- 0 to 2)
      for (j <- 0 to 1)
        assert(G(i, j) == G1(i, j))
  }
}

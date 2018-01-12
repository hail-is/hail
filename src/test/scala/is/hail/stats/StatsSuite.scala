package is.hail.stats

import breeze.linalg.DenseMatrix
import is.hail.SparkSuite
import is.hail.utils._
import is.hail.testUtils._
import is.hail.variant.{Genotype, Locus, Variant}
import org.apache.commons.math3.distribution.{ChiSquaredDistribution, NormalDistribution}
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
    val vds = vdsFromGtMatrix(hc)(G)

    val G1 = DenseMatrix.zeros[Int](3, 2)

    vds.typedRDD[Locus, Variant].collect().foreach{ case (v, (va, gs)) => gs.zipWithIndex.foreach { case (g, i) => G1(i, v.start - 1) = Genotype.gt(g).getOrElse(-1) } }

    assert(vds.sampleIds == IndexedSeq("0", "1", "2"))
    assert(vds.variants.collect().toSet == Set(Variant("1", 1, "A", "C"), Variant("1", 2, "A", "C")))

    for (i <- 0 to 2)
      for (j <- 0 to 1)
        assert(G(i, j) == G1(i, j))
  }

  @Test def poissonTest() {
    // compare with R
    assert(D_==(dpois(5, 10), 0.03783327))
    assert(qpois(0.3, 10) == 8)
    assert(qpois(0.3, 10, lowerTail = false) == 12)
    assert(D_==(ppois(5, 10), 0.06708596))
    assert(D_==(ppois(5, 10, lowerTail = false), 0.932914))
    assert(rpois(5) >= 0)

    assert(qpois(ppois(5, 10), 10) == 5)
    assert(qpois(ppois(5, 10, lowerTail = false), 10, lowerTail = false) == 5)

    assert(ppois(30, 1, lowerTail = false) > 0)
  }

  @Test def betaTest() {
    val tol = 1e-5

    assert(D_==(dbeta(.2 , 1, 3), 1.92, tol))
    assert(D_==(dbeta(0.70, 2, 10), 0.001515591, tol))
    assert(D_==(dbeta(.4, 5, 3), 0.96768, tol))
    assert(D_==(dbeta(.3, 7, 2), 0.0285768, tol))
    assert(D_==(dbeta(.8, 2, 2), .96, tol))
    assert(D_==(dbeta(.1, 3, 6), 0.9920232, tol))
    assert(D_==(dbeta(.6, 3, 4), 1.3824, tol))
    assert(D_==(dbeta(.1, 1, 1), 1, tol))
    assert(D_==(dbeta(.2, 4, 7), 1.761608, tol))
    assert(D_==(dbeta(.2, 1, 2), 1.6, tol))

  }

  @Test def binomTestTest() {
    //Compare against R
    assert(D_==(binomTest(2, 10, 0.5, "two.sided"), 0.10937, tolerance = 1e-4))
    assert(D_==(binomTest(4 ,10, 0.5, "less"), 0.377, tolerance = 1e-3))
    assert(D_==(binomTest(32, 50, 0.4, "greater"), 0.0005193, tolerance = 1e-4))

  }

  @Test def entropyTest() {
    assert(D_==(entropy("accctg"), 1.79248, tolerance = 1e-5))
    assert(D_==(entropy(Array(2, 3, 4, 5, 6, 6, 4)), 2.23593, tolerance = 1e-5))

  }

}

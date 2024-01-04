package is.hail.stats

import is.hail.{HailSuite, TestUtils}
import is.hail.TestUtils._
import is.hail.testUtils._
import is.hail.utils._
import is.hail.variant._

import breeze.linalg.DenseMatrix
import org.apache.commons.math3.distribution.{ChiSquaredDistribution, NormalDistribution}
import org.testng.annotations.Test

class StatsSuite extends HailSuite {

  @Test def chiSquaredTailTest() {
    val chiSq1 = new ChiSquaredDistribution(1)
    assert(D_==(pchisqtail(1d, 1), 1 - chiSq1.cumulativeProbability(1d)))
    assert(D_==(pchisqtail(5.52341d, 1), 1 - chiSq1.cumulativeProbability(5.52341d)))

    val chiSq2 = new ChiSquaredDistribution(2)
    assert(D_==(pchisqtail(1, 2), 1 - chiSq2.cumulativeProbability(1)))
    assert(D_==(pchisqtail(5.52341, 2), 1 - chiSq2.cumulativeProbability(5.52341)))

    val chiSq5 = new ChiSquaredDistribution(5.2)
    assert(D_==(pchisqtail(1, 5.2), 1 - chiSq5.cumulativeProbability(1)))
    assert(D_==(pchisqtail(5.52341, 5.2), 1 - chiSq5.cumulativeProbability(5.52341)))

    assert(D_==(qchisqtail(.1, 1.0), chiSq1.inverseCumulativeProbability(1 - .1)))
    assert(D_==(qchisqtail(.0001, 1.0), chiSq1.inverseCumulativeProbability(1 - .0001)))

    val a = List(.0000000001, .5, .9999999999, 1.0)
    a.foreach(p => assert(D_==(pchisqtail(qchisqtail(p, 1.0), 1.0), p)))

    // compare with R
    assert(math.abs(pchisqtail(400, 1) - 5.507248e-89) < 1e-93)
    assert(D_==(qchisqtail(5.507248e-89, 1), 400))
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

  @Test def poissonTest() {
    // compare with R
    assert(D_==(dpois(5, 10), 0.03783327))
    assert(qpois(0.3, 10) == 8)
    assert(qpois(0.3, 10, lowerTail = false, logP = false) == 12)
    assert(D_==(ppois(5, 10), 0.06708596))
    assert(D_==(ppois(5, 10, lowerTail = false, logP = false), 0.932914))

    assert(qpois(ppois(5, 10), 10) == 5)
    assert(qpois(
      ppois(5, 10, lowerTail = false, logP = false),
      10,
      lowerTail = false,
      logP = false,
    ) == 5)

    assert(ppois(30, 1, lowerTail = false, logP = false) > 0)
  }

  @Test def betaTest() {
    val tol = 1e-5

    assert(D_==(dbeta(.2, 1, 3), 1.92, tol))
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

  @Test def entropyTest() {
    assert(D_==(entropy("accctg"), 1.79248, tolerance = 1e-5))
    assert(D_==(entropy(Array(2, 3, 4, 5, 6, 6, 4)), 2.23593, tolerance = 1e-5))

  }

}

package org.broadinstitute.hail.stats

import org.apache.commons.math3.distribution.ChiSquaredDistribution
import org.broadinstitute.hail.utils._
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
  }
}

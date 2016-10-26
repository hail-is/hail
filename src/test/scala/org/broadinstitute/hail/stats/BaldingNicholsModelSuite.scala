package org.broadinstitute.hail.stats

import breeze.linalg.DenseVector
import org.broadinstitute.hail.SparkSuite
import org.testng.annotations.Test

class BaldingNicholsModelSuite extends SparkSuite {

  @Test def baldingNicholsTest() = {
    val K = 3
    val N = 10
    val M = 100
    val popDist = DenseVector[Double](1d, 2d, 3d)
    val FstOfPop = DenseVector[Double](.1, .2, .3)
    val seed = 0

    val bnm = BaldingNicholsModel(K, N, M, Some(popDist), Some(FstOfPop), Some(seed))
    val bnm1 = BaldingNicholsModel(K, N, M, Some(popDist), Some(FstOfPop), Some(seed))

    assert(bnm.genotypes == bnm1.genotypes)
  }

}

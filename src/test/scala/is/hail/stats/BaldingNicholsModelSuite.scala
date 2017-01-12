package is.hail.stats

import breeze.linalg.DenseVector
import is.hail.SparkSuite
import is.hail.variant.VariantDataset
import org.testng.annotations.Test

class BaldingNicholsModelSuite extends SparkSuite {

  /*@Test def baldingNicholsTest() = {
    val K = 3
    val N = 10
    val M = 100
    val popDist = DenseVector[Double](1d, 2d, 3d)
    val FstOfPop = DenseVector[Double](.1, .2, .3)
    val seed = 0

    val bnm = BaldingNicholsModelDist(sc, K, N, M, popDist, FstOfPop, seed)
    val bnm1 = BaldingNicholsModelDist(sc, K, N, M, popDist, FstOfPop, seed)

    assert()
  }*/

  @Test def testDimensions() = {
    val K = 5
    val N = 10
    val M = 100
    val popDist = DenseVector[Double](1, 2, 3, 4, 5)
    val FstOfPop = DenseVector[Double](.1, .2, .3, .2, .2)
    val seed = 0

    val bnm: VariantDataset = BaldingNicholsModelDist(sc, K, N, M, popDist, FstOfPop, seed)

    //Check right number of samples
    val allCorrectSize = bnm.rdd.collect.map(x => x._2._2).forall(x => x.size == 10)
    assert(allCorrectSize)


    //Check right number of variants
    assert(bnm.rdd.count() == 100)
  }

}

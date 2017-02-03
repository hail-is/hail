package is.hail.stats

import breeze.linalg._
import breeze.stats.mean
import is.hail.utils._
import is.hail.{SparkSuite, TestUtils, stats}
import org.testng.annotations.Test

class ComputeRRMSuite extends SparkSuite {

  @Test def normalizedGtArrayAndComputeRRMTest() {

    //small example
    val G0 = DenseMatrix((0,  1,  0,  2),
                         (2, -1,  0,  2),
                         (0,  2,  0, -1))

    val W = DenseMatrix((0.0, 1.0),
                        (2.0, 1.5),
                        (0.0, 2.0))

    val vds = stats.vdsFromMatrix(hc)(G0)

    val n = vds.nSamples
    val gtVects = vds.rdd.collect().flatMap { case (v, (va, gs)) => toNormalizedGtArray(gs, n) }.map(DenseVector(_))

    for (gts <- gtVects) {
      assert(math.abs(mean(gts)) < 1e-6)
      assert(D_==(norm(gts), math.sqrt(n)))
    }

    for (i <- 0 until W.cols) {
      W(::, i) -= mean(W(::, i))
      W(::, i) *= math.sqrt(n) / norm(W(::, i))
    }
    val Klocal = (W * W.t) / W.cols.toDouble
    val KwithoutBlock = ComputeRRM(vds, useBlock = false)._1
    val KwithBlock = ComputeRRM(vds, useBlock = true)._1

    TestUtils.assertMatrixEqualityDouble(Klocal, KwithoutBlock)
    TestUtils.assertMatrixEqualityDouble(KwithBlock, KwithoutBlock)


    // medium example with many constant gt vectors
    val seed = 0
    val n1 = 100
    val m1 = 200
    val k = 3
    val Fst = .9
    val FstOfPop = Array.fill[Double](k)(Fst)

    val vds1 = BaldingNicholsModel(hc, k, n1, m1, None, Some(FstOfPop), seed, Some(4), UniformDist(0.1, 0.9))

    val G1 = TestUtils.vdsToMatrixInt(vds1)

    val W1 = convert(TestUtils.removeConstantCols(G1), Double)
    for (i <- 0 until W1.cols) {
      W1(::, i) -= mean(W1(::, i))
      W1(::, i) *= math.sqrt(n1) / norm(W1(::, i))
    }

    val K1local = (W1 * W1.t) / W1.cols.toDouble
    val K1withoutBlock = ComputeRRM(vds1, useBlock = false)._1
    val K1withBlock = ComputeRRM(vds1, useBlock = true)._1

    TestUtils.assertMatrixEqualityDouble(K1local, K1withoutBlock)
    TestUtils.assertMatrixEqualityDouble(K1withBlock, K1withoutBlock)
  }
}
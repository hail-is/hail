package is.hail.methods

import breeze.linalg.{DenseMatrix, convert, norm}
import breeze.stats.mean
import is.hail.utils._
import is.hail.{SparkSuite, TestUtils, stats}
import org.testng.Assert
import org.testng.annotations.Test

class LDMatrixSuite extends SparkSuite {

  val m = 100
  val n = 100
  val seed = scala.util.Random.nextInt()
  val vds = hc.baldingNicholsModel(1, n, m, seed = seed)
  val ldMatrix = vds.ldMatrix()

  /**
    * Tests that entries in LDMatrix agree with those computed by LDPrune.computeR. Also tests
    * that index i in the matrix corresponds to index i in the array.
    */
  @Test def testEntries() {
    val nSamples = vds.nSamples

    val variantsTable = vds.rdd.map { case (v, (_, gs)) =>
      (v, LDPrune.toBitPackedVector(gs.hardCallGenotypeIterator, nSamples))}.collectAsMap()


    val indexToBPV = ldMatrix.variants.map(v => variantsTable(v).get)
    val ldMatrixLocal = ldMatrix.matrix.toBlockMatrix().toLocalMatrix()
    val numVariants = ldMatrixLocal.numRows

    for(i <- 0 until numVariants; j <- 0 until numVariants) {
      val computedR = LDPrune.computeR(indexToBPV(i), indexToBPV(j))
      val matrixR = ldMatrixLocal(i, j)

      Assert.assertEquals(computedR, matrixR, .000001)
    }
  }

  /**
    * Tests that variants are ordered in array.
    */
  @Test def testOrderingOfVariants() {
    assert(ldMatrix.variants.isSorted)
  }

  /**
    * Tests that the method agrees with local Breeze math.
    */
  @Test def localTest() {
    val m = 2
    val n = 3
    val genotypes: DenseMatrix[Int] = DenseMatrix((0, 2, 1),
                                                  (1, 0, 0))


    //Normalize genotype matrix
    def localLDCompute(G: DenseMatrix[Int]): DenseMatrix[Double] = {
      val W = convert(G, Double)

      // each row has mean 0, norm sqrt(n), variance 1
      for (i <- 0 until W.cols) {
        W(::, i) -= mean(W(::, i))
        W(::, i) *= math.sqrt(n) / norm(W(::, i))
      }

      val nSamples = W.cols
      val ld = (W * W.t) / nSamples.toDouble
      ld
    }

    val localLD = localLDCompute(genotypes)

    val distLdSpark = stats.vdsFromMatrix(hc)(genotypes.t).ldMatrix().matrix.toBlockMatrix().toLocalMatrix()
    val distLDBreeze = new DenseMatrix[Double](distLdSpark.numRows, distLdSpark.numCols, distLdSpark.toArray)

    println(distLDBreeze)
    println()
    println(localLD)

    TestUtils.assertMatrixEqualityDouble(distLDBreeze, localLD, tolerance = 1)

  }

}

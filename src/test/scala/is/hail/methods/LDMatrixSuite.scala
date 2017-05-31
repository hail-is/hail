package is.hail.methods

import breeze.linalg.{DenseMatrix, convert, norm}
import breeze.stats.mean
import is.hail.utils._
import is.hail.{SparkSuite, TestUtils, stats}
import org.testng.Assert
import org.testng.annotations.Test

class LDMatrixSuite extends SparkSuite {

  val m = 200
  val n = 100
  val seed = scala.util.Random.nextInt()
  lazy val vds = hc.baldingNicholsModel(1, n, m, seed = seed)
  lazy val ldMatrix = vds.ldMatrix(30)
  lazy val localMatrix = ldMatrix.toLocalMatrix()


  @Test def testSymmetry() {
    val numVariants = ldMatrix.variants.length

    for(i <- 0 until numVariants; j <- 0 to i) {
      Assert.assertEquals(localMatrix(i, j), localMatrix(j, i), 0.0000001, s"Failed on seed $seed.")
    }
  }
  /**
    * Tests that entries in LDMatrix agree with those computed by LDPrune.computeR. Also tests
    * that index i in the matrix corresponds to index i in the array.
    */
  @Test def testEntries() {
    val nSamples = vds.nSamples

    val variantsTable = vds.rdd.map { case (v, (_, gs)) =>
      (v, LDPrune.toBitPackedVector(gs.hardCallIterator, nSamples))}.collectAsMap()


    val indexToBPV = ldMatrix.variants.map(v => variantsTable(v).get)
    val numVariants = localMatrix.numRows

    for(i <- 0 until numVariants; j <- 0 until numVariants) {
      val computedR = LDPrune.computeR(indexToBPV(j), indexToBPV(i))
      val matrixR = localMatrix(j, i)

      Assert.assertEquals(computedR, matrixR, .000001)
    }
  }

  /**
    * Test that LD Matrix is square.
    */
  @Test def testDimensions() {
    assert(ldMatrix.matrix.numRows().toInt == ldMatrix.matrix.numCols())
    assert(ldMatrix.matrix.numRows().toInt == ldMatrix.variants.length)
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

    def localLDCompute(G: DenseMatrix[Int]): DenseMatrix[Double] = {
      val W = convert(G, Double).t

      for (i <- 0 until W.cols) {
        W(::, i) -= mean(W(::, i))
        W(::, i) *= math.sqrt(n) / norm(W(::, i))
      }

      val ld = (W.t * W) / n.toDouble
      ld
    }

    val localLD = localLDCompute(genotypes)

    val vds = stats.vdsFromMatrix(hc)(genotypes.t)
    val distLDSpark = vds.ldMatrix(1).matrix.toBlockMatrix().toLocalMatrix()
    val distLDBreeze = new DenseMatrix[Double](distLDSpark.numRows, distLDSpark.numCols, distLDSpark.toArray)

    TestUtils.assertMatrixEqualityDouble(distLDBreeze, localLD)

  }

}

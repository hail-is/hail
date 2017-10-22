package is.hail.methods

import breeze.linalg.{DenseMatrix, convert, norm}
import breeze.stats.mean
import is.hail.utils._
import is.hail.variant.VariantDataset
import is.hail.{SparkSuite, TestUtils, stats}
import org.testng.Assert
import org.testng.annotations.Test

class LDMatrixSuite extends SparkSuite {
  val m = 50
  val n = 60
  val seed: Int = scala.util.Random.nextInt()
  lazy val vds: VariantDataset = hc.baldingNicholsModel(1, n, m, seed = seed)
  lazy val distLDMatrix: LDMatrix = LDMatrix.apply(vds, Some(false))
  lazy val localLDMatrix: LDMatrix = vds.ldMatrix(forceLocal = true)

  /**
    * Tests that entries in LDMatrix agree with those computed by LDPrune.computeR. Also tests
    * that index i in the matrix corresponds to index i in the array.
    */
  @Test def testEntries() {
    val nSamples = vds.nSamples

    val variantsTable = vds.rdd.map { case (v, (_, gs)) =>
      (v, LDPrune.toBitPackedVector(gs.hardCallIterator, nSamples))}.collectAsMap()

    val indexToBPV = distLDMatrix.variants.map(v => variantsTable(v).get)
    val distLDMatrixLocal = distLDMatrix.matrix.toHailBlockMatrix().toLocalMatrix()
    val localLDMatrixLocal = localLDMatrix.matrix.toHailBlockMatrix().toLocalMatrix()
    val numVariants = distLDMatrixLocal.rows

    for(i <- 0 until numVariants; j <- 0 until numVariants) {
      val computedR = LDPrune.computeR(indexToBPV(i), indexToBPV(j))
      val distMatrixR = distLDMatrixLocal(i, j)
      val localMatrixR = localLDMatrixLocal(i, j)

      Assert.assertEquals(computedR, distMatrixR, .000001)
      Assert.assertEquals(computedR, localMatrixR, .000001)
    }
  }

  /**
    * Tests that variants are ordered in array.
    */
  @Test def testOrderingOfVariants() {
    assert(distLDMatrix.variants.isSorted)
    assert(localLDMatrix.variants.isSorted)
  }

  /**
    * Tests that the method agrees with local Breeze math.
    */
  @Test def localTest() {
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

    val vds = stats.vdsFromGtMatrix(hc)(genotypes.t)
    val distLdSpark = vds.ldMatrix().matrix.toBlockMatrix().toLocalMatrix()
    val distLDBreeze = new DenseMatrix[Double](distLdSpark.numRows, distLdSpark.numCols, distLdSpark.toArray)

    TestUtils.assertMatrixEqualityDouble(distLDBreeze, localLD)

  }

  @Test
  def readWriteIdentityLocalLDMatrix() {
    val actual = localLDMatrix

    val fname = tmpDir.createTempFile("test")
    actual.write(fname)
    assert(actual.toLocalMatrix == LDMatrix.read(hc, fname).toLocalMatrix)
  }

  @Test
  def readWriteIdentityDistLDMatrix() {
    val actual = distLDMatrix

    val fname = tmpDir.createTempFile("test")
    actual.write(fname)
    assert(actual.toLocalMatrix == LDMatrix.read(hc, fname).toLocalMatrix)
  }

}

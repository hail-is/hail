package is.hail.methods

import is.hail.utils._
import is.hail.SparkSuite
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

}

package is.hail.methods

import is.hail.SparkSuite
import org.testng.annotations.Test

/**
  * Tests for KinshipMatrix.
  */
class KinshipMatrixSuite extends SparkSuite {

  @Test def testFilterSamples() {
    val km = hc.baldingNicholsModel(1, 15, 15).rrm()
    val kmFilt = km.filterSamples(s => s.toInt < 7 && s.toInt > 3)
    val kmFiltLocalMatrix = kmFilt.matrix.toBlockMatrix().toLocalMatrix()

    assert(kmFilt.sampleIds.length == 3)
    assert(kmFiltLocalMatrix.numCols == 3)
    assert(kmFiltLocalMatrix.numRows == 3)
  }
}

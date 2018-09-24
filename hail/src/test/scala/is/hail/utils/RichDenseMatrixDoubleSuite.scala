package is.hail.utils

import is.hail.{SparkSuite, TestUtils}
import is.hail.utils.richUtils.RichDenseMatrixDouble
import breeze.linalg.{DenseMatrix => BDM}
import is.hail.linalg.BlockMatrix
import org.testng.annotations.Test

class RichDenseMatrixDoubleSuite extends SparkSuite {
  @Test
  def readWriteBDM() {
    val m = BDM.rand[Double](256, 129) // 33024 doubles
    val fname = tmpDir.createTempFile("test")

    m.write(hc, fname, bufferSpec = BlockMatrix.bufferSpec)
    val m2 = RichDenseMatrixDouble.read(hc, fname, BlockMatrix.bufferSpec)

    assert(m === m2)
  }
  
  @Test
  def testReadWriteDoubles(): Unit = {
    val file = tmpDir.createTempFile("test")
    val m = BDM.rand[Double](50, 100)
    RichDenseMatrixDouble.exportToDoubles(hc, file, m, forceRowMajor = false)
    val m2 = RichDenseMatrixDouble.importFromDoubles(hc, file, 50, 100, rowMajor = false)
    assert(m === m2)
    
    val fileT = tmpDir.createTempFile("test2")
    val mT = m.t
    RichDenseMatrixDouble.exportToDoubles(hc, fileT, mT, forceRowMajor = true)
    val lmT2 = RichDenseMatrixDouble.importFromDoubles(hc, fileT, 100, 50, rowMajor = true)
    assert(mT === mT)
    
    TestUtils.interceptFatal("Premature") {
      RichDenseMatrixDouble.importFromDoubles(hc, fileT, 100, 100, rowMajor = true)
    }
  }
}

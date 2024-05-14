package is.hail.utils

import is.hail.{HailSuite, TestUtils}
import is.hail.linalg.BlockMatrix
import is.hail.utils.richUtils.RichDenseMatrixDouble

import breeze.linalg.{DenseMatrix => BDM}
import org.testng.annotations.Test

class RichDenseMatrixDoubleSuite extends HailSuite {
  @Test
  def readWriteBDM(): Unit = {
    val m = BDM.rand[Double](256, 129) // 33024 doubles
    val fname = ctx.createTmpPath("test")

    m.write(fs, fname, bufferSpec = BlockMatrix.bufferSpec)
    val m2 = RichDenseMatrixDouble.read(fs, fname, BlockMatrix.bufferSpec)

    assert(m === m2)
  }

  @Test
  def testReadWriteDoubles(): Unit = {
    val file = ctx.createTmpPath("test")
    val m = BDM.rand[Double](50, 100)
    RichDenseMatrixDouble.exportToDoubles(fs, file, m, forceRowMajor = false)
    val m2 = RichDenseMatrixDouble.importFromDoubles(fs, file, 50, 100, rowMajor = false)
    assert(m === m2)

    val fileT = ctx.createTmpPath("test2")
    val mT = m.t
    RichDenseMatrixDouble.exportToDoubles(fs, fileT, mT, forceRowMajor = true)
    val lmT2 = RichDenseMatrixDouble.importFromDoubles(fs, fileT, 100, 50, rowMajor = true)
    assert(mT === mT)

    TestUtils.interceptFatal("Premature") {
      RichDenseMatrixDouble.importFromDoubles(fs, fileT, 100, 100, rowMajor = true)
    }
  }
}

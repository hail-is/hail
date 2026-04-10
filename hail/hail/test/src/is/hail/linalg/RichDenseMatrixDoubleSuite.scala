package is.hail.linalg

import is.hail.HailSuite
import is.hail.linalg.implicits._

import breeze.linalg.{DenseMatrix => BDM}

class RichDenseMatrixDoubleSuite extends HailSuite {
  test("readWriteBDM") {
    val m = BDM.rand[Double](256, 129) // 33024 doubles
    val fname = ctx.createTmpPath("test")

    m.write(fs, fname, bufferSpec = BlockMatrix.bufferSpec)
    val m2 = RichDenseMatrixDouble.read(fs, fname, BlockMatrix.bufferSpec)

    assertEquals(m, m2)
  }

  test("ReadWriteDoubles") {
    val file = ctx.createTmpPath("test")
    val m = BDM.rand[Double](50, 100)
    RichDenseMatrixDouble.exportToDoubles(fs, file, m, forceRowMajor = false): Unit
    val m2 = RichDenseMatrixDouble.importFromDoubles(fs, file, 50, 100, rowMajor = false)
    assertEquals(m, m2)

    val fileT = ctx.createTmpPath("test2")
    val mT = m.t
    RichDenseMatrixDouble.exportToDoubles(fs, fileT, mT, forceRowMajor = true): Unit
    val lmT2 = RichDenseMatrixDouble.importFromDoubles(fs, fileT, 100, 50, rowMajor = true)
    assertEquals(mT, lmT2)

    interceptFatal("Premature") {
      RichDenseMatrixDouble.importFromDoubles(fs, fileT, 100, 100, rowMajor = true)
    }
  }
}

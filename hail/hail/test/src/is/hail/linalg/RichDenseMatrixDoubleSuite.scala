package is.hail.linalg

import is.hail.TestUtils._
import is.hail.backend.ExecuteContext
import is.hail.linalg.implicits._

import breeze.linalg.{DenseMatrix => BDM}
import org.junit.jupiter.api.Test

class RichDenseMatrixDoubleSuite {
  @Test
  def readWriteBDM(implicit ctx: ExecuteContext): Unit = {
    val fs = ctx.fs
    val m = BDM.rand[Double](256, 129) // 33024 doubles
    val fname = ctx.createTmpPath("test")

    m.write(fs, fname, bufferSpec = BlockMatrix.bufferSpec)
    val m2 = RichDenseMatrixDouble.read(fs, fname, BlockMatrix.bufferSpec)

    assertEq(m, m2)
  }

  @Test
  def testReadWriteDoubles(implicit ctx: ExecuteContext): Unit = {
    val fs = ctx.fs
    val file = ctx.createTmpPath("test")
    val m = BDM.rand[Double](50, 100)
    RichDenseMatrixDouble.exportToDoubles(fs, file, m, forceRowMajor = false): Unit
    val m2 = RichDenseMatrixDouble.importFromDoubles(fs, file, 50, 100, rowMajor = false)
    assertEq(m, m2)

    val fileT = ctx.createTmpPath("test2")
    val mT = m.t
    RichDenseMatrixDouble.exportToDoubles(fs, fileT, mT, forceRowMajor = true): Unit
    val lmT2 = RichDenseMatrixDouble.importFromDoubles(fs, fileT, 100, 50, rowMajor = true)
    assertEq(mT, lmT2)

    interceptFatal("Premature") {
      RichDenseMatrixDouble.importFromDoubles(fs, fileT, 100, 100, rowMajor = true)
    }
  }
}

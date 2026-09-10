package is.hail.linalg

import is.hail.TestUtils.{assertEq, intercept, interceptFatal}
import is.hail.backend.ExecuteContext

import org.junit.jupiter.api.Test

// Guards against breeze 2.1.0's broken `DenseMatrix.isContiguous`: slices of
// square matrices are falsely contiguous, so breeze's `dst := src` and
// `dst := x` vectorise over raw arrays and corrupt data. See
// is.hail.linalg.DenseMatrix.
class DenseMatrixSuite {

  private def square: DenseMatrix =
    DenseMatrix.tabulate(3, 3)((i, j) => (i * 3 + j).toDouble)

  @Test def copyIntoRowSliceOfSquareMatrix(): Unit = {
    val dst = DenseMatrix.zeros(3, 3)
    dst(0 until 2, ::) := square(0 until 2, 0 until 3): Unit
    assertEq(dst, DenseMatrix((0.0, 1.0, 2.0), (3.0, 4.0, 5.0), (0.0, 0.0, 0.0)))
  }

  @Test def copyIntoOffsetRowSliceOfSquareMatrix(): Unit = {
    val dst = DenseMatrix.zeros(3, 3)
    dst(2 until 3, ::) := square(1 until 2, 0 until 3): Unit
    assertEq(dst, DenseMatrix((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (3.0, 4.0, 5.0)))
  }

  @Test def copyFromTransposedView(): Unit = {
    val dst = DenseMatrix.zeros(3, 3)
    dst(0 until 2, ::) := square.t(0 until 2, 0 until 3): Unit
    assertEq(dst, DenseMatrix((0.0, 3.0, 6.0), (1.0, 4.0, 7.0), (0.0, 0.0, 0.0)))
  }

  @Test def copyBetweenTransposedViews(): Unit = {
    val dst = DenseMatrix.zeros(3, 3)
    dst.t(0 until 2, ::) := square.t(0 until 2, 0 until 3): Unit
    assertEq(dst, DenseMatrix((0.0, 1.0, 0.0), (3.0, 4.0, 0.0), (6.0, 7.0, 0.0)))
  }

  // breeze's `copy` is `fresh := this` and so looks gated on the broken
  // `isContiguous`, but a falsely-contiguous view is never square while a
  // fresh compact destination only passes the check when square, so the
  // corrupt fast path cannot fire; these pin that delegating `copy` is safe
  @Test def copyOfRowSliceOfSquareMatrix(): Unit =
    assertEq(square(0 until 2, ::).copy, DenseMatrix((0.0, 1.0, 2.0), (3.0, 4.0, 5.0)))

  @Test def copyOfColSliceOfTransposedSquareMatrix(): Unit =
    assertEq(square.t(::, 0 until 2).copy, DenseMatrix((0.0, 3.0), (1.0, 4.0), (2.0, 5.0)))

  @Test def fillRowSliceOfSquareMatrix(): Unit = {
    val dst = square
    dst(1 until 2, ::) := 0.0: Unit
    assertEq(dst, DenseMatrix((0.0, 1.0, 2.0), (0.0, 0.0, 0.0), (6.0, 7.0, 8.0)))
  }

  @Test def fillRowSliceOfTransposedSquareMatrix(): Unit = {
    val dst = square.t
    dst(1 until 2, ::) := 0.0: Unit
    assertEq(dst, DenseMatrix((0.0, 3.0, 6.0), (0.0, 0.0, 0.0), (2.0, 5.0, 8.0)))
  }

  // a reversed full slice is a negative-stride view over the whole backing
  // array, so a size check alone would pass it off as compact
  @Test def toArrayOfReversedColSliceOfSquareMatrix(): Unit =
    assert(square(::, 2 to 0 by -1).toArray
      sameElements Array(2.0, 5.0, 8.0, 1.0, 4.0, 7.0, 0.0, 3.0, 6.0))

  // breeze has no minor stride: a stepped slice along the minor axis is
  // rejected, so `mapArray` and `:=` only need `majorStride`
  @Test def steppedRowSliceOfSquareMatrixThrows(): Unit =
    intercept[IllegalArgumentException](square(0 to 2 by 2, ::)): Unit

  // a stepped slice along the major axis folds the step into `majorStride`
  @Test def toArrayOfSteppedColSliceOfSquareMatrix(): Unit =
    assert(square(::, 0 to 2 by 2).toArray sameElements Array(0.0, 3.0, 6.0, 2.0, 5.0, 8.0))

  @Test def copyIntoSteppedColSliceOfSquareMatrix(): Unit = {
    val dst = DenseMatrix.zeros(3, 3)
    dst(::, 0 to 2 by 2) := DenseMatrix((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)): Unit
    assertEq(dst, DenseMatrix((1.0, 0.0, 2.0), (3.0, 0.0, 4.0), (5.0, 0.0, 6.0)))
  }

  @Test def copyFromSteppedColSliceOfSquareMatrix(): Unit = {
    val dst = DenseMatrix.zeros(3, 2)
    dst := square(::, 0 to 2 by 2): Unit
    assertEq(dst, DenseMatrix((0.0, 2.0), (3.0, 5.0), (6.0, 8.0)))
  }

  @Test def copyIntoReversedColSliceOfSquareMatrix(): Unit = {
    val dst = DenseMatrix.zeros(3, 3)
    dst(::, 2 to 0 by -1) := square: Unit
    assertEq(dst, DenseMatrix((2.0, 1.0, 0.0), (5.0, 4.0, 3.0), (8.0, 7.0, 6.0)))
  }

  @Test def fillSteppedColSliceOfSquareMatrix(): Unit = {
    val dst = square
    dst(::, 0 to 2 by 2) := 0.0: Unit
    assertEq(dst, DenseMatrix((0.0, 1.0, 0.0), (0.0, 4.0, 0.0), (0.0, 7.0, 0.0)))
  }

  @Test def mapOverReversedColSliceOfSquareMatrix(): Unit =
    assertEq(
      square(::, 2 to 0 by -1).map(identity),
      DenseMatrix((2.0, 1.0, 0.0), (5.0, 4.0, 3.0), (8.0, 7.0, 6.0)),
    )

  @Test def readWrite(implicit ctx: ExecuteContext): Unit = {
    val fs = ctx.fs
    val m = DenseMatrix.rand(256, 129) // 33024 doubles
    val fname = ctx.createTmpPath("test")

    m.write(fs, fname, bufferSpec = BlockMatrix.bufferSpec)
    val m2 = DenseMatrix.read(fs, fname, BlockMatrix.bufferSpec)

    assertEq(m, m2)
  }

  @Test def readWriteDoubles(implicit ctx: ExecuteContext): Unit = {
    val fs = ctx.fs
    val file = ctx.createTmpPath("test")
    val m = DenseMatrix.rand(50, 100)
    DenseMatrix.exportToDoubles(fs, file, m, forceRowMajor = false): Unit
    val m2 = DenseMatrix.importFromDoubles(fs, file, 50, 100, rowMajor = false)
    assertEq(m, m2)

    val fileT = ctx.createTmpPath("test2")
    val mT = m.t
    DenseMatrix.exportToDoubles(fs, fileT, mT, forceRowMajor = true): Unit
    val lmT2 = DenseMatrix.importFromDoubles(fs, fileT, 100, 50, rowMajor = true)
    assertEq(mT, lmT2)

    interceptFatal("Premature") {
      DenseMatrix.importFromDoubles(fs, fileT, 100, 100, rowMajor = true)
    }
  }
}

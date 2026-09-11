package is.hail

import is.hail.TestUtils._
import is.hail.linalg.DenseMatrix

import breeze.linalg.DenseVector
import org.junit.jupiter.api.Test

class TestUtilsSuite {

  @Test def matrixEqualityTest(): Unit = {
    val M = DenseMatrix((1d, 0d), (0d, 1d))
    val M1 = DenseMatrix((1d, 0d), (0d, 1.0001d))
    val V = DenseVector(0d, 1d)
    val V1 = DenseVector(0d, 0.5d)

    assertMatrixEqualityDouble(M, DenseMatrix.tabulate(2, 2)((i, j) => if (i == j) 1d else 0d))
    assertMatrixEqualityDouble(M, M1, 0.001)
    assertVectorEqualityDouble(V, 2d * V1)

    intercept[AssertionError](assertVectorEqualityDouble(V, V1)): Unit
    intercept[AssertionError](assertMatrixEqualityDouble(M, M1)): Unit
  }

  @Test def constantVectorTest(): Unit = {
    assert(isConstant(DenseVector()))
    assert(isConstant(DenseVector(0)))
    assert(isConstant(DenseVector(0, 0)))
    assert(isConstant(DenseVector(0, 0, 0)))
    assert(!isConstant(DenseVector(0, 1)))
    assert(!isConstant(DenseVector(0, 0, 1)))
  }
}

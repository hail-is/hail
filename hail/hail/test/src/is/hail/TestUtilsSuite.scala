package is.hail

import breeze.linalg.{DenseMatrix, DenseVector}
import org.scalatest
import org.testng.annotations.Test

class TestUtilsSuite extends HailSuite {

  @Test def matrixEqualityTest(): scalatest.Assertion = {
    val M = DenseMatrix((1d, 0d), (0d, 1d))
    val M1 = DenseMatrix((1d, 0d), (0d, 1.0001d))
    val V = DenseVector(0d, 1d)
    val V1 = DenseVector(0d, 0.5d)

    assertMatrixEqualityDouble(M, DenseMatrix.eye(2))
    assertMatrixEqualityDouble(M, M1, 0.001)
    assertVectorEqualityDouble(V, 2d * V1)

    assertThrows[Exception](assertVectorEqualityDouble(V, V1))
    assertThrows[Exception](assertMatrixEqualityDouble(M, M1))
  }

  @Test def constantVectorTest(): scalatest.Assertion = {
    assert(isConstant(DenseVector()))
    assert(isConstant(DenseVector(0)))
    assert(isConstant(DenseVector(0, 0)))
    assert(isConstant(DenseVector(0, 0, 0)))
    assert(!isConstant(DenseVector(0, 1)))
    assert(!isConstant(DenseVector(0, 0, 1)))
  }

  @Test def removeConstantColsTest(): scalatest.Assertion = {
    val M = DenseMatrix((0, 0, 1, 1, 0), (0, 1, 0, 1, 1))

    val M1 = DenseMatrix((0, 1, 0), (1, 0, 1))

    assert(removeConstantCols(M) == M1)
  }
}

package is.hail

import java.io.File
import java.lang.reflect.Modifier
import java.net.URI

import breeze.linalg.{DenseMatrix, DenseVector}
import org.testng.annotations.{DataProvider, Test}

class TestUtilsSuite extends HailSuite {

  @Test def matrixEqualityTest() {
    val M = DenseMatrix((1d, 0d), (0d, 1d))
    val M1 = DenseMatrix((1d, 0d), (0d, 1.0001d))
    val V = DenseVector(0d, 1d)
    val V1 = DenseVector(0d, 0.5d)

    TestUtils.assertMatrixEqualityDouble(M, DenseMatrix.eye(2))
    TestUtils.assertMatrixEqualityDouble(M, M1, 0.001)
    TestUtils.assertVectorEqualityDouble(V, 2d * V1)

    intercept[Exception](TestUtils.assertVectorEqualityDouble(V, V1))
    intercept[Exception](TestUtils.assertMatrixEqualityDouble(M, M1))
  }

  @Test def constantVectorTest() {
    assert(TestUtils.isConstant(DenseVector()))
    assert(TestUtils.isConstant(DenseVector(0)))
    assert(TestUtils.isConstant(DenseVector(0, 0)))
    assert(TestUtils.isConstant(DenseVector(0, 0, 0)))
    assert(!TestUtils.isConstant(DenseVector(0, 1)))
    assert(!TestUtils.isConstant(DenseVector(0, 0, 1)))
  }

  @Test def removeConstantColsTest(): Unit = {
    val M = DenseMatrix((0, 0, 1, 1, 0),
                        (0, 1, 0, 1, 1))

    val M1 = DenseMatrix((0, 1, 0),
                         (1, 0, 1))

    assert(TestUtils.removeConstantCols(M) == M1)
  }
}

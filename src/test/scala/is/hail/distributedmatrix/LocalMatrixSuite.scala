package is.hail.distributedmatrix

import breeze.linalg.{DenseMatrix => BDM}
import is.hail.{SparkSuite, TestUtils}
import org.testng.annotations.Test

class LocalMatrixSuite extends SparkSuite {
  def assertEqual(lm1: LocalMatrix, lm2: LocalMatrix) { assert(lm1.m === lm2.m) }
  def assertApproxEqual(lm1: LocalMatrix, lm2: LocalMatrix) { TestUtils.assertMatrixEqualityDouble(lm1.m, lm2.m) }
  
  @Test
  def writeRead() {
    
  }
  
  @Test
  def ops() {    
    import LocalMatrix.ops._
    
    val e = 2.0
    val x = LocalMatrix(1, 1, Array(2.0))
    val c = LocalMatrix(2, 1, Array(1.0, 2.0))
    val r = LocalMatrix(1, 3, Array(1.0, 2.0, 3.0))
    val m = LocalMatrix(2, 3, Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    
    // add    
    assertEqual(x + x, e * x)
    assertEqual(c + c, e * c)
    assertEqual(r + r, e * r)
    assertEqual(m + m, e * m)

    assertEqual(x + c, LocalMatrix(2, 1, Array(3.0, 4.0)))
    assertEqual(x + c, c + x)
    assertEqual(x + r, LocalMatrix(1, 3, Array(3.0, 4.0, 5.0)))
    assertEqual(x + r, r + x)
    assertEqual(x + m, LocalMatrix(2, 3, Array(3.0, 4.0, 5.0, 6.0, 7.0, 8.0)))
    assertEqual(x + m, m + x)
    assertEqual(x + m, e + m)
    assertEqual(x + m, m + e)
    
    assertEqual(c + m, LocalMatrix(2, 3, Array(2.0, 4.0, 4.0, 6.0, 6.0, 8.0)))
    assertEqual(c + m, m + c)
    assertEqual(r + m, LocalMatrix(2, 3, Array(2.0, 3.0, 5.0, 6.0, 8.0, 9.0)))
    assertEqual(r + m, m + r)
    assertEqual(c + r, LocalMatrix(2, 3, Array(2.0, 3.0, 3.0, 4.0, 4.0, 5.0)))
    assertEqual(c + r, r + c)
    
    // subtract
    assertEqual((x + x) - x, x)
    assertEqual((c + c) - c, c)
    assertEqual((r + r) - r, r)
    assertEqual((m + m) - m, m)

    assertEqual(x - c, LocalMatrix(2, 1, Array(1.0, 0.0)))
    assertEqual(x - c, (c - x).negation())
    assertEqual(x - r, LocalMatrix(1, 3, Array(1.0, 0.0, -1.0)))
    assertEqual(x - r, (r - x).negation())
    assertEqual(x - m, LocalMatrix(2, 3, Array(1.0, 0.0, -1.0, -2.0, -3.0, -4.0)))
    assertEqual(x - m, (m - x).negation())
    assertEqual(x - m, e - m)
    assertEqual(x - m, (m - e).negation())
    
    assertEqual(c - m, LocalMatrix(2, 3, Array(0.0, 0.0, -2.0, -2.0, -4.0, -4.0)))
    assertEqual(c - m, (m - c).negation())
    assertEqual(r - m, LocalMatrix(2, 3, Array(0.0, -1.0, -1.0, -2.0, -2.0, -3.0)))
    assertEqual(r - m, (m - r).negation())
    assertEqual(c - r, LocalMatrix(2, 3, Array(0.0, 1.0, -1.0, 0.0, -2.0, -1.0)))
    assertEqual(c - r, (r - c).negation())
    
    // multiply
    assertEqual(x * x, LocalMatrix(1, 1, Array(4.0)))
    assertEqual(c * c, LocalMatrix(2, 1, Array(1.0, 4.0)))
    assertEqual(r * r, LocalMatrix(1, 3, Array(1.0, 4.0, 9.0)))
    assertEqual(m * m, LocalMatrix(2, 3, Array(1.0, 4.0, 9.0, 16.0, 25.0, 36.0)))

    assertEqual(x * c, LocalMatrix(2, 1, Array(2.0, 4.0)))
    assertEqual(x * c, c * x)
    assertEqual(x * r, LocalMatrix(1, 3, Array(2.0, 4.0, 6.0)))
    assertEqual(x * r, r * x)
    assertEqual(x * m, LocalMatrix(2, 3, Array(2.0, 4.0, 6.0, 8.0, 10.0, 12.0)))
    assertEqual(x * m, m * x)
    assertEqual(x * m, e * m)
    assertEqual(x * m, m * e)
    
    assertEqual(c * m, LocalMatrix(2, 3, Array(1.0, 4.0, 3.0, 8.0, 5.0, 12.0)))
    assertEqual(c * m, m * c)
    assertEqual(r * m, LocalMatrix(2, 3, Array(1.0, 2.0, 6.0, 8.0, 15.0, 18.0)))
    assertEqual(r * m, m * r)
    assertEqual(c * r, LocalMatrix(2, 3, Array(1.0, 2.0, 2.0, 4.0, 3.0, 6.0)))    
    
    // divide
    assertApproxEqual((x * x) / x, x)
    assertApproxEqual((c * c) / c, c)
    assertApproxEqual((r * r) / r, r)
    assertApproxEqual((m * m) / m, m)

    assertApproxEqual(x / c, LocalMatrix(2, 1, Array(2.0, 1.0)))
    assertApproxEqual(x / c, (c / x).reciprocal())
    assertApproxEqual(x / r, LocalMatrix(1, 3, Array(2.0, 1.0, 2.0 / 3)))
    assertApproxEqual(x / r, (r / x).reciprocal())
    assertApproxEqual(x / m, LocalMatrix(2, 3, Array(2.0, 1.0, 2.0 / 3, 0.5, 0.4, 1.0 / 3)))
    assertApproxEqual(x / m, (m / x).reciprocal())
    assertApproxEqual(x / m, e / m)
    assertApproxEqual(x / m, (m / e).reciprocal())
    
    assertApproxEqual(c / m, LocalMatrix(2, 3, Array(1.0, 1.0, 1.0 / 3, 0.5, 0.2, 1.0 / 3)))
    assertApproxEqual(c / m, (m / c).reciprocal())
    assertApproxEqual(r / m, LocalMatrix(2, 3, Array(1.0, 0.5, 2.0 / 3, 0.5, 0.6, 0.5)))
    assertApproxEqual(r / m, (m / r).reciprocal())
    assertApproxEqual(c / r, LocalMatrix(2, 3, Array(1.0, 2.0, 0.5, 1.0, 1.0 / 3, 2.0 / 3)))
    assertApproxEqual(c / r, (r / c).reciprocal())
  }
}

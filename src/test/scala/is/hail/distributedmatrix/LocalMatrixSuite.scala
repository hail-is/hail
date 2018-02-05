package is.hail.distributedmatrix

import is.hail.{SparkSuite, TestUtils}
import org.testng.annotations.Test
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}

class LocalMatrixSuite extends SparkSuite {
  
  def assertEqual(lm1: LocalMatrix, lm2: LocalMatrix) { assert(lm1.m === lm2.m) }
  def assertApproxEqual(lm1: LocalMatrix, lm2: LocalMatrix) { TestUtils.assertMatrixEqualityDouble(lm1.m, lm2.m) }
  
  @Test
  def applyWriteRead() {
    val fname = tmpDir.createTempFile("test")
    
    val m1 = LocalMatrix(2, 3, Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    val m2 = LocalMatrix(2, 3, Array(1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 5.0, 6.0, 0.0), 0, 3, isTransposed = false)
    val m3 = LocalMatrix(2, 3, Array(0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 5.0, 6.0), 1, 3, isTransposed = false)
    val m4 = LocalMatrix(2, 3, Array(1.0, 3.0, 5.0, 2.0, 4.0, 6.0), isTransposed = true)
    val m5 = LocalMatrix(2, 3, Array(0.0, 1.0, 3.0, 5.0, 0.0, 2.0, 4.0, 6.0), 1, 4, isTransposed = true)
    
    for { m <- Seq(m1, m2, m3, m4, m5) } {
      m.write(hc, fname)
      assertEqual(LocalMatrix.read(hc, fname), m1)
    }
    
    val v1 = LocalMatrix(Array(1.0, 2.0))
    val v2 = LocalMatrix(BDV[Double](1.0, 2.0))
    val v3 = LocalMatrix(new BDV[Double](Array(0.0, 1.0, 0.0, 2.0, 0.0, 0.0), offset = 1, stride = 2, length = 2))

    for { v <- Seq(v1, v2, v3) } {
      v.write(hc, fname)
      assertEqual(LocalMatrix.read(hc, fname), v1)
    }
  }
  
  @Test
  def checkShapesTest() {
    import TestUtils.interceptFatal
    
    val x = LocalMatrix(1, 1, Array(2.0))
    val c = LocalMatrix(2, 1, Array(1.0, 2.0))
    val r = LocalMatrix(1, 3, Array(1.0, 2.0, 3.0))
    val m = LocalMatrix(2, 3, Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))    

    LocalMatrix.checkShapes(m, m, "")
    LocalMatrix.checkShapes(m, r, "")
    LocalMatrix.checkShapes(m, c, "")
    LocalMatrix.checkShapes(m, x, "")
    LocalMatrix.checkShapes(r, m, "")
    LocalMatrix.checkShapes(r, r, "")
    LocalMatrix.checkShapes(r, c, "")
    LocalMatrix.checkShapes(r, x, "")
    LocalMatrix.checkShapes(c, m, "")
    LocalMatrix.checkShapes(c, r, "")
    LocalMatrix.checkShapes(c, c, "")
    LocalMatrix.checkShapes(c, x, "")
    LocalMatrix.checkShapes(x, m, "")
    LocalMatrix.checkShapes(x, r, "")
    LocalMatrix.checkShapes(x, c, "")
    LocalMatrix.checkShapes(x, x, "")    
    
    interceptFatal("Incompatible shapes") { LocalMatrix.checkShapes(m.t, m, "") }
    interceptFatal("Incompatible shapes") { LocalMatrix.checkShapes(m.t, r, "") }
    interceptFatal("Incompatible shapes") { LocalMatrix.checkShapes(m.t, c, "") }
    interceptFatal("Incompatible shapes") { LocalMatrix.checkShapes(m, m.t, "") }
    interceptFatal("Incompatible shapes") { LocalMatrix.checkShapes(m, r.t, "") }
    interceptFatal("Incompatible shapes") { LocalMatrix.checkShapes(m, c.t, "") }
    interceptFatal("Incompatible shapes") { LocalMatrix.checkShapes(r.t, m, "") }
    interceptFatal("Incompatible shapes") { LocalMatrix.checkShapes(r.t, c, "") }
    interceptFatal("Incompatible shapes") { LocalMatrix.checkShapes(r, m.t, "") }
    interceptFatal("Incompatible shapes") { LocalMatrix.checkShapes(r, c.t, "") }
    interceptFatal("Incompatible shapes") { LocalMatrix.checkShapes(c.t, m, "") }
    interceptFatal("Incompatible shapes") { LocalMatrix.checkShapes(c.t, r, "") }
    interceptFatal("Incompatible shapes") { LocalMatrix.checkShapes(c, m.t, "") }
    interceptFatal("Incompatible shapes") { LocalMatrix.checkShapes(c, r.t, "") }
  }
  
  @Test
  def ops() {
    import LocalMatrix.ops._
    import TestUtils.interceptFatal
    
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
    
    interceptFatal("addition") { m.t + m }

    // subtract
    assertEqual((x + x) - x, x)
    assertEqual((c + c) - c, c)
    assertEqual((r + r) - r, r)
    assertEqual((m + m) - m, m)

    assertEqual(x - c, LocalMatrix(2, 1, Array(1.0, 0.0)))
    assertEqual(x - c, -(c - x))
    assertEqual(x - r, LocalMatrix(1, 3, Array(1.0, 0.0, -1.0)))
    assertEqual(x - r, -(r - x))
    assertEqual(x - m, LocalMatrix(2, 3, Array(1.0, 0.0, -1.0, -2.0, -3.0, -4.0)))
    assertEqual(x - m, -(m - x))
    assertEqual(x - m, e - m)
    assertEqual(x - m, -(m - e))
    
    assertEqual(c - m, LocalMatrix(2, 3, Array(0.0, 0.0, -2.0, -2.0, -4.0, -4.0)))
    assertEqual(c - m, -(m - c))
    assertEqual(r - m, LocalMatrix(2, 3, Array(0.0, -1.0, -1.0, -2.0, -2.0, -3.0)))
    assertEqual(r - m, -(m - r))
    assertEqual(c - r, LocalMatrix(2, 3, Array(0.0, 1.0, -1.0, 0.0, -2.0, -1.0)))
    assertEqual(c - r, -(r - c))
    
    interceptFatal("subtraction") { m.t - m }
    
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

    interceptFatal("multiplication") { m.t * m }
    
    // divide
    assertApproxEqual((x * x) / x, x)
    assertApproxEqual((c * c) / c, c)
    assertApproxEqual((r * r) / r, r)
    assertApproxEqual((m * m) / m, m)

    assertApproxEqual(x / c, LocalMatrix(2, 1, Array(2.0, 1.0)))
    assertApproxEqual(x / c, 1.0 / (c / x))
    assertApproxEqual(x / r, LocalMatrix(1, 3, Array(2.0, 1.0, 2.0 / 3)))
    assertApproxEqual(x / r, 1.0 / (r / x))
    assertApproxEqual(x / m, LocalMatrix(2, 3, Array(2.0, 1.0, 2.0 / 3, 0.5, 0.4, 1.0 / 3)))
    assertApproxEqual(x / m, 1.0 / (m / x))
    assertApproxEqual(x / m, e / m)
    assertApproxEqual(x / m, 1.0 / (m / e))
    
    assertApproxEqual(c / m, LocalMatrix(2, 3, Array(1.0, 1.0, 1.0 / 3, 0.5, 0.2, 1.0 / 3)))
    assertApproxEqual(c / m, 1.0 / (m / c))
    assertApproxEqual(r / m, LocalMatrix(2, 3, Array(1.0, 0.5, 2.0 / 3, 0.5, 0.6, 0.5)))
    assertApproxEqual(r / m, 1.0 / (m / r))
    assertApproxEqual(c / r, LocalMatrix(2, 3, Array(1.0, 2.0, 0.5, 1.0, 1.0 / 3, 2.0 / 3)))
    assertApproxEqual(c / r, 1.0 / (r / c))
    
    interceptFatal("division") { m.t / m }    
  }
}

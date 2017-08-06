package is.hail.stats

import breeze.linalg._
import breeze.stats.mean
import is.hail.SparkSuite
import is.hail.stats
import org.testng.annotations.Test

class EigendecompositionSuite extends SparkSuite {

  def compareEigen(e1: Eigendecomposition, e2: Eigendecomposition, tolerance: Double = 1e-6) {
    assert(e1.rowSignature == e2.rowSignature)
    assert(e1.rowIds sameElements e2.rowIds)
    compareArray(e1.evals.toArray, e2.evals.toArray)
    compareArray(e1.evects(::, 1 to -1).toArray, e2.evects(::, 1 to -1).toArray)
  }
  
  // comparing up to element sign
  def compareArray(a1: Array[Double], a2: Array[Double], tolerance: Double = 1e-6) {
    assert((a1, a2).zipped.forall((x, y) => math.abs(x - y) < tolerance || math.abs(x + y) < tolerance))
  }
  
  // comparison over non-zero eigenvalues
  @Test def test() {
    def testMatrix(G: DenseMatrix[Int], H: DenseMatrix[Double]) {
      for (i <- 0 until H.cols) {
        H(::, i) -= mean(H(::, i))
        H(::, i) *= math.sqrt(H.rows) / norm(H(::, i))
      }
      val K = (1.0 / H.cols) * (H * H.t)
      val L = (1.0 / H.rows) * (H.t * H)

      val eigen = eigSymD(K)
      val rank = H.rows min H.cols - 1
      
      val vds = stats.vdsFromGtMatrix(hc)(G)
      val eigenK = vds.rrm().eigen(None)
      val eigenL = vds.ldMatrix().eigenRRM(vds, None)
      
      compareArray(eigen.eigenvalues(-rank to -1).toArray, eigenK.evals(-rank to -1).toArray)
      compareArray(eigen.eigenvectors(::, -rank to -1).toArray, eigenK.evects(::, -rank to -1).toArray)
      compareEigen(eigenK, eigenL)
    }
    
    val G = DenseMatrix((0, 1),
                        (2, 1),
                        (0, 2))
        
    val G1 = DenseMatrix((0,  1,  0,  2),
                         (2, -1,  0,  2),
                         (1,  2,  0, -1))
    
    val H1 = DenseMatrix((0.0, 1.0),
                         (2.0, 1.5),
                         (1.0, 2.0))

    val G2 = DenseMatrix((0, 1, 2),
                         (2, 1, 0),
                         (0, 2, 1))
    
    val G3 = DenseMatrix((0, 1, 2, 1, 0),
                         (2, 1, 0, 2, 1),
                         (0, 2, 0, 0, 0))

    
    testMatrix(G, convert(G, Double))
    testMatrix(G1, H1)
    testMatrix(G2, convert(G2, Double))
    testMatrix(G3, convert(G3, Double))
  }
}
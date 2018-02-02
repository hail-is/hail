package is.hail.distributedmatrix

import breeze.linalg.{DenseMatrix => BDM, _}
import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import is.hail.HailContext
import is.hail.utils.richUtils.RichDenseMatrixDouble
import is.hail.utils._
import org.apache.commons.math3.random.MersenneTwister

object LocalMatrix {
  def apply(m: BDM[Double]) = new LocalMatrix(m)

  def apply(nRows: Int, nCols: Int, data: Array[Double]) {
    BDM(nRows, nCols, data)
  }
  
  def apply(nRows: Int, nCols: Int, data: Array[Double], isTransposed: Boolean) {
    BDM(nRows, nCols, data, isTransposed)
  }
  
  def read(hc: HailContext, path: String) {
    val m = RichDenseMatrixDouble.read(hc, path)
    new LocalMatrix(m)
  }
  
  def random(nRows: Int, nCols: Int, seed: Int = 0) {
    val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
    BDM.rand(nRows, nCols, randBasis.uniform)
  }
  
  def randomGaussian(nRows: Int, nCols: Int, seed: Int = 0) {
    val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
    BDM.rand(nRows, nCols, randBasis.gaussian)
  }
}

// wrapper for Breeze DenseMatrix[Double] with zero offset and minimal stride
class LocalMatrix(m: BDM[Double]) {
  require(m.offset == 0)
  require(m.majorStride == (if (m.isTranspose) m.rows else m.cols))
  
  def write(hc: HailContext, path: String) {
    m.write(hc: HailContext, path)
  }
}

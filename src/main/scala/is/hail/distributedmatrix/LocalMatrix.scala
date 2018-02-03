package is.hail.distributedmatrix

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import is.hail.HailContext
import is.hail.utils.richUtils.RichDenseMatrixDouble
import is.hail.utils._
import org.apache.commons.math3.random.MersenneTwister

object LocalMatrix {
  type M = LocalMatrix  
  
  def apply(m: BDM[Double]): M = new LocalMatrix(m)

  def apply(nRows: Int, nCols: Int): M =
    LocalMatrix(new BDM[Double](nRows, nCols))
  
  def apply(nRows: Int, nCols: Int, data: Array[Double]): M =
    LocalMatrix(new BDM(nRows, nCols, data))
  
  def apply(nRows: Int, nCols: Int, data: Array[Double], isTransposed: Boolean): M = {
    val m = new BDM[Double](nRows, nCols, data, 0, if (isTransposed) nCols else nRows, isTransposed)
    LocalMatrix(m)
  }
  
  def read(hc: HailContext, path: String): M = new LocalMatrix(RichDenseMatrixDouble.read(hc, path))
  
  def random(nRows: Int, nCols: Int, seed: Int = 0, gaussian: Boolean = false): M = {
    val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
    val rand = if (gaussian) randBasis.gaussian else randBasis.uniform
    LocalMatrix(BDM.rand(nRows, nCols, rand))
  }
  
  object ops {

    implicit class Shim(l: M) {
      def t: M =
        l.transpose()

      def diag: Array[Double] =
        l.diagonal()

      def *(r: M): M =
        l.multiply(r)

      def :+(r: M): M =
        l.add(r)

      def :-(r: M): M =
        l.subtract(r)

      def :*(r: M): M =
        l.pointwiseMultiply(r)

      def :/(r: M): M =
        l.pointwiseDivide(r)

      def +(r: Double): M =
        l.scalarAdd(r)

      def -(r: Double): M =
        l.scalarSubtract(r)

      def *(r: Double): M =
        l.scalarMultiply(r)

      def /(r: Double): M =
        l.scalarDivide(r)

      def :+(v: Array[Double]): M =
        l.vectorAddToEveryColumn(v)

      def :*(v: Array[Double]): M =
        l.vectorPointwiseMultiplyEveryColumn(v)

      def --+(v: Array[Double]): M =
        l.vectorAddToEveryRow(v)

      def --*(v: Array[Double]): M =
        l.vectorPointwiseMultiplyEveryRow(v)
    }

    implicit class ScalarShim(l: Double) {
      def +(r: M): M =
        r.scalarAdd(l)

      def -(r: M): M =
        r.scalarSubtractLeft(l)

      def *(r: M): M =
        r.scalarMultiply(l)

      def /(r: M): M =
        r.scalarDivideLeft(l)
    }
  }
}

// wrapper for Breeze DenseMatrix[Double] with zero offset and minimal stride
class LocalMatrix(val m: BDM[Double]) {
  require(m.offset == 0)
  require(m.majorStride == (if (m.isTranspose) m.rows else m.cols))

  def nRows: Int = m.rows

  def nCols: Int = m.cols

  def data: Array[Double] = m.data

  def toArray: Array[Double] = m.toArray
  
  def isTranspose: Boolean = m.isTranspose

  def write(hc: HailContext, path: String) { m.write(hc: HailContext, path) }

  def toBlockMatrix(hc: HailContext, blockSize: Int = BlockMatrix.defaultBlockSize): BlockMatrix =
    BlockMatrix.from(hc.sc, m, blockSize)
  
  def transpose() = LocalMatrix(m.t)

  def diagonal(): Array[Double] = diag(m).toArray

  def add(that: LocalMatrix) = LocalMatrix(m + that.m)

  def subtract(that: LocalMatrix) = LocalMatrix(m - that.m)

  def multiply(that: LocalMatrix) = LocalMatrix(m * that.m)

  def pointwiseMultiply(that: LocalMatrix) = LocalMatrix(m :* that.m)

  def pointwiseDivide(that: LocalMatrix) = LocalMatrix(m :/ that.m)

  def scalarAdd(i: Double) = LocalMatrix(m + i)

  def scalarSubtract(i: Double) = LocalMatrix(m - i)
  
  def scalarSubtractLeft(i: Double) = LocalMatrix(i - m)

  def scalarMultiply(i: Double) = LocalMatrix(m :* i)

  def scalarDivide(i: Double) = LocalMatrix(m :/ i)  
  
  def scalarDivideLeft(i: Double) = LocalMatrix(i :/ m)

  def vectorAddToEveryRow(v: Array[Double]) = LocalMatrix(m(*, ::) + BDV(v))  
  
  def vectorAddToEveryColumn(v: Array[Double]) = LocalMatrix(m(::, *) + BDV(v))

  def vectorPointwiseMultiplyEveryRow(v: Array[Double]) = LocalMatrix(m(*, ::) :* BDV(v))  
  
  def vectorPointwiseMultiplyEveryColumn(v: Array[Double]) = LocalMatrix(m(::, *) :* BDV(v))
}

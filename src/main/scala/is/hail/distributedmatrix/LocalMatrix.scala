package is.hail.distributedmatrix

import is.hail.HailContext
import is.hail.stats.eigSymD
import is.hail.utils.richUtils.RichDenseMatrixDouble
import is.hail.utils._
import breeze.linalg.{inv, DenseMatrix => BDM, DenseVector => BDV, svd => breezeSVD, * => B_*, _}
import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import org.apache.commons.math3.random.MersenneTwister


object LocalMatrix {
  type M = LocalMatrix

  private val sclrType = 0
  private val colType = 1
  private val rowType = 2
  private val matType = 3

  def apply(m: BDM[Double]): M = new LocalMatrix(m)

  // vector => matrix with single column
  def apply(v: BDV[Double]): M = LocalMatrix(new BDM(v.length, 1, v.toArray))

  def zeros(nRows: Int, nCols: Int): M =
    LocalMatrix(new BDM[Double](nRows, nCols))

  def apply(nRows: Int, nCols: Int, data: Array[Double]): M =
    LocalMatrix(new BDM(nRows, nCols, data))

  def apply(nRows: Int, nCols: Int, data: Array[Double], isTransposed: Boolean): M = {
    val m = new BDM[Double](nRows, nCols, data, 0, if (isTransposed) nCols else nRows, isTransposed)
    LocalMatrix(m)
  }

  def apply(a: Array[Double]): M = {
    require(a.nonEmpty)
    LocalMatrix(a.length, 1, a)
  }

  def read(hc: HailContext, path: String): M = new LocalMatrix(RichDenseMatrixDouble.read(hc, path))

  def random(nRows: Int, nCols: Int, seed: Int = 0, gaussian: Boolean = false): M = {
    val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
    val rand = if (gaussian) randBasis.gaussian else randBasis.uniform
    LocalMatrix(BDM.rand(nRows, nCols, rand))
  }

  def outerSum(row: Array[Double], col: Array[Double]): LocalMatrix = {
    val nRows = col.length
    val nCols = row.length
    assert(nRows > 0 && nCols > 0 && (nRows * nCols.toLong < Int.MaxValue))
    
    val a = new Array[Double](nRows * nCols)
      var j = 0
      while (j < nCols) {
        var i = 0
        while (i < nRows) {
          a(j * nRows + i) = col(i) + row(j)
          i += 1
        }
        j += 1
      }
      LocalMatrix(nRows, nCols, a)
  }

  object ops {
    implicit class ScalarShim(l: Double) {
      def +(r: M): M = LocalMatrix(l :+ r.m)

      def -(r: M): M = LocalMatrix(l :- r.m)

      def *(r: M): M = LocalMatrix(l :* r.m)

      def /(r: M): M = LocalMatrix(l :/ r.m)
    }
  }
}

// Matrix with NumPy-style ops and broadcasting
class LocalMatrix(val m: BDM[Double]) {
  def nRows: Int = m.rows

  def nCols: Int = m.cols

  def shape: (Int, Int) = (nRows, nCols)

  def isTranspose: Boolean = m.isTranspose  

  def data: Array[Double] = m.data

  def toArray: Array[Double] =
    if (m.isCompact && !m.isTranspose)
      m.data
    else
      m.toArray

  def copy() = LocalMatrix(m.copy)

  def apply(i: Int, j: Int) = m(i, j)

  def write(hc: HailContext, path: String) {
    m.write(hc: HailContext, path)
  }

  def toBlockMatrix(hc: HailContext, blockSize: Int = BlockMatrix.defaultBlockSize): BlockMatrix =
    BlockMatrix.from(hc.sc, m, blockSize)

  def +(e: Double) = LocalMatrix(m :+ e)

  def -(e: Double) = LocalMatrix(m :- e)

  def *(e: Double) = LocalMatrix(m :* e)

  def /(e: Double) = LocalMatrix(m :/ e) 

  def unary_+ = this

  def unary_- = LocalMatrix(-m)

  import LocalMatrix.{sclrType, colType, rowType, matType}
  import LocalMatrix.ops._

  private def shapeType: Int =
    if (nRows > 1) {
      if (nCols > 1) matType else colType
    } else {
      if (nCols > 1) rowType else sclrType
    }

  def +(that: LocalMatrix): LocalMatrix =
    if (shapeType == that.shapeType)
      LocalMatrix(m + that.m)
    else if (that.shapeType == sclrType)
      this + that(0, 0)
    else  if (shapeType == sclrType)
      that + m(0, 0)
    else
      (shapeType, that.shapeType) match {
        case (`matType`, `colType`) => LocalMatrix(m(::, B_*) :+ BDV(that.data))
        case (`matType`, `rowType`) => LocalMatrix(m(B_*, ::) :+ BDV(that.data))
        case (`colType`, `matType`) => LocalMatrix(that.m(::, B_*) :+ BDV(data))
        case (`rowType`, `matType`) => LocalMatrix(that.m(B_*, ::) :+ BDV(data))
        case (`colType`, `rowType`) => LocalMatrix.outerSum(that.data, data)
        case (`rowType`, `colType`) => LocalMatrix.outerSum(data, that.data)
      }

  def -(that: LocalMatrix): LocalMatrix =
    if (shapeType == that.shapeType)
      LocalMatrix(m - that.m)
    else if (that.shapeType == sclrType)
      this - that(0, 0)
    else if (shapeType == sclrType)
      m(0, 0) - that
    else
      (shapeType, that.shapeType) match {
        case (`matType`, `colType`) => LocalMatrix(m(::, B_*) :- BDV(that.data))
        case (`matType`, `rowType`) => LocalMatrix(m(B_*, ::) :- BDV(that.data))
        case _ => this + (-that) // FIXME: room for improvement
      }

  // pointwise multiplication
  def *(that: LocalMatrix): LocalMatrix =
    if (shapeType == that.shapeType)
      LocalMatrix(m :* that.m)
    else if (that.shapeType == sclrType)
      this * that(0, 0)
    else if (shapeType == sclrType)
      that * m(0, 0)
    else
      (shapeType, that.shapeType) match {
        case (`matType`, `colType`) => LocalMatrix(m(::, B_*) :* BDV(that.data))
        case (`matType`, `rowType`) => LocalMatrix(m(B_*, ::) :* BDV(that.data))
        case (`colType`, `matType`) => LocalMatrix(that.m(::, B_*) :* BDV(data))
        case (`rowType`, `matType`) => LocalMatrix(that.m(B_*, ::) :* BDV(data))
        case (`colType`, `rowType`) => LocalMatrix(m * that.m)
        case (`rowType`, `colType`) => LocalMatrix(that.m * m)
      }

  // pointwise division
  def /(that: LocalMatrix): LocalMatrix =
    if (shapeType == that.shapeType)
      LocalMatrix(m :/ that.m)
    else if (that.shapeType == sclrType)
      this / that(0, 0)
    else if (shapeType == sclrType)
      m(0, 0) / that
    else
      (shapeType, that.shapeType) match {
        case (`matType`, `colType`) => LocalMatrix(m(::, B_*) :/ BDV(that.data))
        case (`matType`, `rowType`) => LocalMatrix(m(B_*, ::) :/ BDV(that.data))
        case _ => this * (1.0 / that) // FIXME: room for improvement
      }

  def transpose() = LocalMatrix(m.t)

  def diagonal(): LocalMatrix = LocalMatrix(diag(m).toArray)  

  // matrix inverse
  def inverse() = LocalMatrix(inv(m))

  // matrix multiplication
  def dot(that: LocalMatrix) = LocalMatrix(m * that.m)

  // solve for X in this * X = that
  def solve(that: LocalMatrix) = LocalMatrix(m \ that.m)  

  // eigendecomposition of symmetric matrix using lapack.dsyevd (Divide and Conquer)
  //   returns (eigenvalues, Some(U)) or (eigenvalues, None)
  //   no symmetry check, uses lower triangle only
  def eig(computeEigenvectors: Boolean = true): (LocalMatrix, Option[LocalMatrix]) = {
    val (evals, optEvects) = eigSymD.doeigSymD(m, computeEigenvectors)
    (LocalMatrix(evals), optEvects.map(LocalMatrix(_)))
  }

  // singular value decomposition of n x m matrix using lapack.dgesdd (Divide and Conquer)
  //   returns (U, singular values, V^T) with the following dimensions
  //   regular:        n x n,  min(n,m) x 1,         m x m
  //   reduced: n x min(n,m),  min(n,m) x 1,  min(n,m) x m
  def svd(reduced: Boolean = true): (LocalMatrix, LocalMatrix, LocalMatrix) = {
    val s = if (reduced) breezeSVD.reduced(m) else breezeSVD(m)
    (LocalMatrix(s.leftVectors), LocalMatrix(s.singularValues), LocalMatrix(s.rightVectors))
  }
}

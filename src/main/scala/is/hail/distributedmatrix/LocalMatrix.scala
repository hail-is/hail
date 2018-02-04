package is.hail.distributedmatrix

import is.hail.HailContext
import is.hail.stats.eigSymD
import is.hail.utils.richUtils.RichDenseMatrixDouble
import is.hail.utils._
import breeze.linalg.{inv, DenseMatrix => BDM, DenseVector => BDV, svd => breezeSVD, _}
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
  
  def addOuter(row: Array[Double], col: Array[Double]): BDM[Double] = {
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
      new BDM[Double](nRows, nCols, a)
  }
  
  object ops {

    implicit class Shim(l: M) {
      def +(r: M): M =
        l.add(r)

      def -(r: M): M =
        l.subtract(r)

      def *(r: M): M =
        l.multiply(r)

      def /(r: M): M =
        l.divide(r)

      def +(r: Double): M =
        l.add(r)

      def -(r: Double): M =
        l.subtract(r)

      def *(r: Double): M =
        l.multiply(r)

      def /(r: Double): M =
        l.divide(r)
    }

    implicit class ScalarShim(l: Double) {
      def +(r: M): M =
        r.add(l)

      def -(r: M): M =
        r.subtractLeft(l)

      def *(r: M): M =
        r.multiply(l)

      def /(r: M): M =
        r.divideLeft(l)
    }
  }
}

// wrapper for Breeze DenseMatrix[Double] with zero offset and minimal stride
class LocalMatrix(val m: BDM[Double]) {
  require(m.offset == 0)
  require(m.majorStride == (if (m.isTranspose) m.cols else m.rows))

  def nRows: Int = m.rows

  def nCols: Int = m.cols

  def shape: (Int, Int) = (nRows, nCols)

  def isTranspose: Boolean = m.isTranspose  
  
  def data: Array[Double] = m.data

  def dataColMajor: Array[Double] = if (!isTranspose) m.data else m.toArray

  def toArray: Array[Double] = m.toArray

  def copy() = LocalMatrix(m.copy)
  
  def apply(i: Int, j: Int) = m(i, j)
  
  def write(hc: HailContext, path: String) {
    m.write(hc: HailContext, path)
  }

  def toBlockMatrix(hc: HailContext, blockSize: Int = BlockMatrix.defaultBlockSize): BlockMatrix =
    BlockMatrix.from(hc.sc, m, blockSize)

  def t = LocalMatrix(m.t)

  def diagonal(): LocalMatrix = LocalMatrix(diag(m).toArray)

  def negation() = LocalMatrix(-m)

  def reciprocal() = LocalMatrix(1.0 :/ m)

  def inverse() = LocalMatrix(inv(m))
  
  def dot(that: LocalMatrix) = LocalMatrix(m * that.m)

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

  def add(e: Double) = LocalMatrix(m + e)

  def subtract(e: Double) = LocalMatrix(m - e)
  def subtractLeft(e: Double) = LocalMatrix(e - m)
  
  def multiply(e: Double) = LocalMatrix(m :* e)

  def divide(e: Double) = LocalMatrix(m :/ e)
  def divideLeft(e: Double) = LocalMatrix(e :/ m)  
  
  import LocalMatrix.sclrType, LocalMatrix.colType, LocalMatrix.rowType, LocalMatrix.matType

  private def shapeType: Int =
    if (nRows > 1) {
      if (nCols > 1) matType else colType
    } else {
      if (nCols > 1) rowType else sclrType
    }
  
  def add(that: LocalMatrix) = LocalMatrix(
    if (shapeType == that.shapeType)
      m + that.m
    else if (that.shapeType == sclrType)
      m + that(0, 0)
    else if (shapeType == sclrType)
      that.m + m(0, 0)
    else
      (shapeType, that.shapeType) match {
        case (`matType`, `colType`) => m(::, *) :+ BDV(that.data)
        case (`matType`, `rowType`) => m(*, ::) :+ BDV(that.data)
        case (`colType`, `matType`) => that.m(::, *) :+ BDV(data)
        case (`rowType`, `matType`) => that.m(*, ::) :+ BDV(data)
        case (`colType`, `rowType`) => LocalMatrix.addOuter(that.data, data)
        case (`rowType`, `colType`) => LocalMatrix.addOuter(data, that.data)
      }
  )
  
  def subtract(that: LocalMatrix) = LocalMatrix(
    if (shapeType == that.shapeType)
      m - that.m
    else if (that.shapeType == sclrType)
      m - that(0, 0)
    else if (shapeType == sclrType)
      m(0, 0) - that.m
    else
      (shapeType, that.shapeType) match {
        case (`matType`, `colType`) => m(::, *) :- BDV(that.data)
        case (`matType`, `rowType`) => m(*, ::) :- BDV(that.data)
        case _ => add(that.negation()).m // FIXME: room for improvement
      }
  )

  def multiply(that: LocalMatrix) = LocalMatrix(
    if (shapeType == that.shapeType)
      m :* that.m
    else if (that.shapeType == sclrType)
      m * that(0, 0)
    else if (shapeType == sclrType)
      that.m * m(0, 0)
    else
      (shapeType, that.shapeType) match {
        case (`matType`, `colType`) => m(::, *) :* BDV(that.data)
        case (`matType`, `rowType`) => m(*, ::) :* BDV(that.data)
        case (`colType`, `matType`) => that.m(::, *) :* BDV(data)
        case (`rowType`, `matType`) => that.m(*, ::) :* BDV(data)
        case (`colType`, `rowType`) => m * that.m
        case (`rowType`, `colType`) => that.m * m
      }
  )
  
  def divide(that: LocalMatrix) = LocalMatrix(
    if (shapeType == that.shapeType)
      m :/ that.m
    else if (that.shapeType == sclrType)
      m :/ that(0, 0)
    else if (shapeType == sclrType)
      m(0, 0) :/ that.m
    else
      (shapeType, that.shapeType) match {
        case (`matType`, `colType`) => m(::, *) :/ BDV(that.data)
        case (`matType`, `rowType`) => m(*, ::) :/ BDV(that.data)
        case _ => multiply(that.reciprocal()).m // FIXME: room for improvement
      }
  )
}

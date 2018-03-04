package is.hail.linalg

import is.hail.HailContext
import is.hail.stats.eigSymD
import is.hail.utils.richUtils.RichDenseMatrixDouble
import is.hail.utils._

import scala.collection.immutable.Range
import breeze.linalg.{* => B_*, DenseMatrix => BDM, DenseVector => BDV, cholesky => breezeCholesky, qr => breezeQR, svd => breezeSVD, _}
import breeze.numerics.{pow => breezePow, sqrt => breezeSqrt}
import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import is.hail.io._
import org.apache.commons.math3.random.MersenneTwister


object LocalMatrix {
  type M = LocalMatrix
  
  private val bufferSpec: BufferSpec =
    new BlockingBufferSpec(32 * 1024,
      new LZ4BlockBufferSpec(32 * 1024,
        new StreamBlockBufferSpec))

  private val sclrType = 0
  private val colType = 1
  private val rowType = 2
  private val matType = 3

  def apply(m: BDM[Double]): M = new LocalMatrix(m)

  // vector => matrix with single column
  def apply(v: BDV[Double]): M = {
    val data = if (v.length == v.data.length) v.data else v.toArray
    LocalMatrix(data)
  }

  // array => matrix with single column
  def apply(data: Array[Double]): M = {
    require(data.length > 0)
    LocalMatrix(data.length, 1, data)
  }

  def zeros(nRows: Int, nCols: Int): M =
    LocalMatrix(new BDM[Double](nRows, nCols))

  def apply(nRows: Int, nCols: Int, data: Array[Double], isTransposed: Boolean = false): M = {
    val m = new BDM[Double](nRows, nCols, data, 0, if (isTransposed) nCols else nRows, isTransposed)
    LocalMatrix(m)
  }

  def apply(nRows: Int, nCols: Int, data: Array[Double], isTransposed: Boolean, offset: Int, majorStride: Int): M = {
    val m = new BDM[Double](nRows, nCols, data, offset, majorStride, isTransposed)
    LocalMatrix(m)
  }
  
  def read(hc: HailContext, path: String): M =
    new LocalMatrix(RichDenseMatrixDouble.read(hc, path, LocalMatrix.bufferSpec))
  
  def random(nRows: Int, nCols: Int, seed: Int = 0, uniform: Boolean = true): M = {
    val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
    val rand = if (uniform) randBasis.uniform else randBasis.gaussian
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
    implicit class Shim(l: M) {
      def +(r: M): M = l.add(r)

      def -(r: M): M = l.subtract(r)

      def *(r: M): M = l.multiply(r)

      def /(r: M): M = l.divide(r)
      
      def +(r: Double): M = l.add(r)

      def -(r: Double): M = l.subtract(r)

      def *(r: Double): M = l.multiply(r)

      def /(r: Double): M = l.divide(r)   
    }
    
    implicit class ScalarShim(l: Double) {      
      def +(r: M): M = r + l

      def -(r: M): M = r.rsubtract(l)

      def *(r: M): M = r * l

      def /(r: M): M = r.rdivide(l)
    }
  }
  
  def checkShapes(m1: LocalMatrix, m2: LocalMatrix, op: String): (Int, Int) = {
    val shapeTypes = (m1.shapeType, m2.shapeType)

    val compatible = shapeTypes match {
      case (`matType`, `matType`) => m1.nRows == m2.nRows && m1.nCols == m2.nCols
      case (`matType`, `rowType`) => m1.nCols == m2.nCols
      case (`matType`, `colType`) => m1.nRows == m2.nRows
      case (`rowType`, `matType`) => m1.nCols == m2.nCols
      case (`rowType`, `rowType`) => m1.nCols == m2.nCols
      case (`colType`, `matType`) => m1.nRows == m2.nRows
      case (`colType`, `colType`) => m1.nRows == m2.nRows
      case _ => true
    }
    
    if (!compatible)
      fatal(s"Incompatible shapes for $op with broadcasting: ${ m1.shape } and ${ m2.shape }")
      
    shapeTypes
  }
}

// Matrix with NumPy-style ops and broadcasting
class LocalMatrix(val m: BDM[Double]) {
  val nRows: Int = m.rows

  val nCols: Int = m.cols

  def shape: (Int, Int) = (nRows, nCols)

  val isTranspose: Boolean = m.isTranspose

  def asArray: Array[Double] =
    if (m.isCompact && !isTranspose)
      m.data
    else
      toArray

  def toArray: Array[Double] = m.toArray

  def copy() = LocalMatrix(m.copy)

  def write(hc: HailContext, path: String) {
    m.write(hc, path, bufferSpec = LocalMatrix.bufferSpec)
  }
  
  // writeBlockMatrix and BlockMatrix.read is safer than toBlockMatrix
  def writeBlockMatrix(hc: HailContext, path: String, blockSize: Int = BlockMatrix.defaultBlockSize,
    forceRowMajor: Boolean = false) {
    m.writeBlockMatrix(hc, path, blockSize, forceRowMajor)
  }

  def toBlockMatrix(hc: HailContext, blockSize: Int = BlockMatrix.defaultBlockSize): BlockMatrix =
    BlockMatrix.from(hc.sc, m, blockSize)

  def apply(i: Int, j: Int): Double = m(i, j)

  def apply(i: Int, jj: Range) = LocalMatrix(m(i to i, jj))

  def apply(ii: Range, j: Int) = LocalMatrix(m(ii, j to j))

  def apply(ii: Range, jj: Range) = LocalMatrix(m(ii, jj))

  def add(e: Double) = LocalMatrix(m + e)

  def subtract(e: Double) = LocalMatrix(m - e)
  def rsubtract(e: Double) = LocalMatrix(e - m)
  
  def multiply(e: Double) = LocalMatrix(m * e)

  def divide(e: Double) = LocalMatrix(m / e)
  def rdivide(e: Double) = LocalMatrix(e / m)

  def unary_+ = this

  def unary_- = LocalMatrix(-m)

  import LocalMatrix.{sclrType, colType, rowType, matType}
  import LocalMatrix.ops._

  private val shapeType: Int =
    if (nRows > 1) {
      if (nCols > 1) matType else colType
    } else {
      if (nCols > 1) rowType else sclrType
    }

  def add(that: LocalMatrix): LocalMatrix = {
    val (st, st2) = LocalMatrix.checkShapes(this, that, "addition")
    if (st == st2)
      LocalMatrix(m + that.m)
    else if (st2 == sclrType)
      this + that(0, 0)
    else if (st == sclrType)
      that + m(0, 0)
    else
      (st, st2) match {
        case (`matType`, `colType`) => LocalMatrix(m(::, B_*) +:+ BDV(that.asArray))
        case (`matType`, `rowType`) => LocalMatrix(m(B_*, ::) +:+ BDV(that.asArray))
        case (`colType`, `matType`) => LocalMatrix(that.m(::, B_*) +:+ BDV(this.asArray))
        case (`rowType`, `matType`) => LocalMatrix(that.m(B_*, ::) +:+ BDV(this.asArray))
        case (`colType`, `rowType`) => LocalMatrix.outerSum(row = that.asArray, col = this.asArray)
        case (`rowType`, `colType`) => LocalMatrix.outerSum(row = this.asArray, col = that.asArray)
      }
  }

  def subtract(that: LocalMatrix): LocalMatrix = {
    val (st, st2) = LocalMatrix.checkShapes(this, that, "subtraction")
    if (st == st2)
      LocalMatrix(m - that.m)
    else if (st2 == sclrType)
      this - that(0, 0)
    else if (st == sclrType)
      m(0, 0) - that
    else
      (st, st2) match {
        case (`matType`, `colType`) => LocalMatrix(m(::, B_*) -:- BDV(that.asArray))
        case (`matType`, `rowType`) => LocalMatrix(m(B_*, ::) -:- BDV(that.asArray))
        case _ => this + (-that) // FIXME: room for improvement
      }
  }

  def multiply(that: LocalMatrix): LocalMatrix = {
    val (st, st2) = LocalMatrix.checkShapes(this, that, "pointwise multiplication")
    if (st == st2)
      LocalMatrix(m *:* that.m)
    else if (st2 == sclrType)
      this * that(0, 0)
    else if (st == sclrType)
      that * m(0, 0)
    else
      (shapeType, that.shapeType) match {
        case (`matType`, `colType`) => LocalMatrix(m(::, B_*) *:* BDV(that.asArray))
        case (`matType`, `rowType`) => LocalMatrix(m(B_*, ::) *:* BDV(that.asArray))
        case (`colType`, `matType`) => LocalMatrix(that.m(::, B_*) *:* BDV(this.asArray))
        case (`rowType`, `matType`) => LocalMatrix(that.m(B_*, ::) *:* BDV(this.asArray))
        case (`colType`, `rowType`) => LocalMatrix(m * that.m)
        case (`rowType`, `colType`) => LocalMatrix(that.m * m)
      }
  }

  def divide(that: LocalMatrix): LocalMatrix = {
    val (st, st2) = LocalMatrix.checkShapes(this, that, "pointwise division")
    if (st == st2)
      LocalMatrix(m /:/ that.m)
    else if (st2 == sclrType)
      this / that(0, 0)
    else if (st == sclrType)
      m(0, 0) / that
    else
      (st, st2) match {
        case (`matType`, `colType`) => LocalMatrix(m(::, B_*) /:/ BDV(that.asArray))
        case (`matType`, `rowType`) => LocalMatrix(m(B_*, ::) /:/ BDV(that.asArray))
        case _ => this * (1.0 / that) // FIXME: room for improvement
      }
  }

  def sqrt(): LocalMatrix = LocalMatrix(breezeSqrt(m))

  def pow(e: Double): LocalMatrix = LocalMatrix(breezePow(m, e))

  def t = LocalMatrix(m.t)

  def diagonal(): LocalMatrix = LocalMatrix(diag(m).toArray)

  def inverse() = LocalMatrix(inv(m))

  def matrixMultiply(that: LocalMatrix) = LocalMatrix(m * that.m)

  // solve for X in AX = B, with A = this and B = that
  def solve(that: LocalMatrix) = LocalMatrix(m \ that.m)

  // eigendecomposition of symmetric matrix using lapack.dsyevd (Divide and Conquer)
  //   X = USU^T with U orthonormal and S diagonal
  //   returns (diag(S), U)
  //   no symmetry check, uses lower triangle only
  def eigh(): (LocalMatrix, LocalMatrix) = {
    val (evals, optEvects) = eigSymD.doeigSymD(m, rightEigenvectors = true)
    (LocalMatrix(evals), LocalMatrix(optEvects.get))
  }
  
  def eigvalsh(): LocalMatrix = {
    val (evals, _) = eigSymD.doeigSymD(m, rightEigenvectors = false)
    LocalMatrix(evals)
  }

  // singular value decomposition of n x m matrix X using lapack.dgesdd (Divide and Conquer)
  //   X = USV^T with U and V orthonormal and S diagonal
  //   returns (U, singular values, V^T) with the following dimensions where k = min(n, m)
  //   regular: n x n,  k x 1,  m x m
  //   reduced: n x k,  k x 1,  k x m
  def svd(reduced: Boolean = false): (LocalMatrix, LocalMatrix, LocalMatrix) = {
    val res = if (reduced) breezeSVD.reduced(m) else breezeSVD(m)
    (LocalMatrix(res.leftVectors), LocalMatrix(res.singularValues), LocalMatrix(res.rightVectors))
  }
  
  // QR decomposition of n x m matrix X
  //   X = QR with Q orthonormal columns and R upper triangular
  //   returns (Q, R) with the following dimensions where k = min(n, m)
  //   regular: n x m,  m x m
  //   reduced: n x k,  k x m
  def qr(reduced: Boolean = false): (LocalMatrix, LocalMatrix) = {
    val res = if (reduced) breezeQR.reduced(m) else breezeQR(m)
    (LocalMatrix(res.q), LocalMatrix(res.r))
  }
  
  // Cholesky factor L of a symmetric, positive-definite matrix X
  //   X = LL^T with L lower triangular
  def cholesky(): LocalMatrix = {
    m.forceSymmetry() // needed to prevent failure of symmetry check, even though lapack only uses lower triangle
    LocalMatrix(breezeCholesky(m))
  }
}

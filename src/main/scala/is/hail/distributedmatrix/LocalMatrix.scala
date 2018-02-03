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
  
  def zeros(nRows: Int, nCols: Int): M =
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
        }
      }
      new BDM[Double](nRows, nCols, a)
  }
  
  object ops {

    implicit class Shim(l: M) {
      def t: M =
        l.transpose()

      def diag: Array[Double] =
        l.diagonal()

      def +(r: M): M =
        l.add(r)

      def -(r: M): M =
        l.subtract(r)

      def *(r: M): M =
        l.pointwiseMultiply(r)

      def /(r: M): M =
        l.pointwiseDivide(r)

      def +(r: Double): M =
        l.scalarAdd(r)

      def -(r: Double): M =
        l.scalarSubtract(r)

      def *(r: Double): M =
        l.scalarMultiply(r)

      def /(r: Double): M =
        l.scalarDivide(r)
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

  def dataColMajor: Array[Double] = if (!isTranspose) m.data else m.toArray

  def toArray: Array[Double] = m.toArray

  def copy() = LocalMatrix(m.copy)

  def isTranspose: Boolean = m.isTranspose

  def write(hc: HailContext, path: String) {
    m.write(hc: HailContext, path)
  }

  def toBlockMatrix(hc: HailContext, blockSize: Int = BlockMatrix.defaultBlockSize): BlockMatrix =
    BlockMatrix.from(hc.sc, m, blockSize)

  def transpose() = LocalMatrix(m.t)

  def diagonal(): Array[Double] = diag(m).toArray

  def shape: (Int, Int) = (nRows, nCols)

  def apply(i: Int, j: Int) = m(i, j)

  // 0 if scalar, 1 if single column, 2 if single row, 3 otherwise
  def typ: Int =
    if (nRows > 1) {
      if (nCols > 1) 3 else 1
    } else {
      if (nCols > 1) 2 else 0
    }

  def negate() = LocalMatrix(-m)
  
  def reciprocate() = LocalMatrix(1.0 :/ m)
 
  def dot(that: LocalMatrix) = LocalMatrix(m * that.m)

  def add(that: LocalMatrix) = LocalMatrix(
    if (typ == that.typ)
      m + that.m
    else if (that.typ == 0)
      m + that(0, 0)
    else if (typ == 0)
      that.m + m(0, 0)
    else
      (typ, that.typ) match {
        case (3, 1) => m(*, ::) :+ BDV(that.data)
        case (3, 2) => m(::, *) :+ BDV(that.data)
        case (1, 3) => that.m(*, ::) :+ BDV(data)
        case (2, 3) => that.m(::, *) :+ BDV(data)
        case (1, 2) => LocalMatrix.addOuter(that.data, data)
        case (2, 1) => LocalMatrix.addOuter(data, that.data)
      }
  )
  
  def subtract(that: LocalMatrix) = LocalMatrix(
    if (typ == that.typ)
      m - that.m
    else if (that.typ == 0)
      m - that(0, 0)
    else if (typ == 0)
      m(0, 0) - that.m
    else
      (typ, that.typ) match {
        case (3, 1) => m(*, ::) :- BDV(that.data)
        case (3, 2) => m(::, *) :- BDV(that.data)
        case _ => add(that.negate()).m // FIXME: room for improvement
      }
  )

  def pointwiseMultiply(that: LocalMatrix) = LocalMatrix(
    if (typ == that.typ)
      m :* that.m
    else if (that.typ == 0)
      m * that(0, 0)
    else if (typ == 0)
      that.m * m(0, 0)
    else
      (typ, that.typ) match {
        case (3, 1) => m(*, ::) :* BDV(that.data)
        case (3, 2) => m(::, *) :* BDV(that.data)
        case (1, 3) => that.m(*, ::) :* BDV(data)
        case (2, 3) => that.m(::, *) :* BDV(data)
        case (1, 2) => m * that.m
        case (2, 1) => that.m * m
      }
  )
  
  def pointwiseDivide(that: LocalMatrix) = LocalMatrix(
    if (typ == that.typ)
      m :/ that.m
    else if (that.typ == 0)
      m :/ that(0, 0)
    else if (typ == 0)
      m(0, 0) :/ that.m
    else
      (typ, that.typ) match {
        case (3, 1) => m(*, ::) :/ BDV(that.data)
        case (3, 2) => m(::, *) :/ BDV(that.data)
        case _ => pointwiseMultiply(that.reciprocate()).m // FIXME: room for improvement
      }
  )
  
  def scalarAdd(i: Double) = LocalMatrix(m + i)

  def scalarSubtract(i: Double) = LocalMatrix(m - i)
  
  def scalarSubtractLeft(i: Double) = LocalMatrix(i - m)

  def scalarMultiply(i: Double) = LocalMatrix(m :* i)

  def scalarDivide(i: Double) = LocalMatrix(m :/ i)  
  
  def scalarDivideLeft(i: Double) = LocalMatrix(i :/ m)
}

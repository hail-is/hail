package is.hail.linalg

import is.hail.io._
import is.hail.io.fs.FS
import is.hail.linalg.DenseMatrix.zeros
import is.hail.utils._

import java.io.{DataInputStream, InputStream, OutputStream}
import java.util

import breeze.generic.{ElementwiseUFunc, UFunc}
import breeze.linalg.{eigSym, svd, DenseMatrix => BDM, DenseVector => BDV, Vector => BVector}
import breeze.linalg.support.{CanTraverseValues, LiteralRow}
import breeze.linalg.support.CanTraverseValues.ValuesVisitor
import breeze.stats.distributions.Rand

// Wraps breeze's DenseMatrix[Double] rather than using it directly: as of
// breeze 2.0, `DenseMatrix.isContiguous` compares `majorStride` against the
// wrong dimension, so views like slices of square matrices are falsely
// contiguous and the operations gated on contiguity (`:=`, `toArray`, `map`,
// `horzcat`, ...) silently read and write them as flat arrays. Those are
// hand-rolled below; everything else delegates to breeze.
final class DenseMatrix(private[linalg] val m: BDM[Double]) extends AnyVal {

  import DenseMatrix.{BroadcastedColumns, BroadcastedRows}

  def rows: Int = m.rows
  def cols: Int = m.cols
  def size: Int = rows * cols

  def data: Array[Double] = m.data
  def isTranspose: Boolean = m.isTranspose

  def apply(i: Int, j: Int): Double = m(i, j)
  def update(i: Int, j: Int, x: Double): Unit = m(i, j) = x
  def copy: DenseMatrix = new DenseMatrix(m.copy)

  // breeze's Matrix.iterator visits elements through `apply`, so it is safe
  // on views
  def iterator: Iterator[((Int, Int), Double)] = m.iterator

  // slices are views of this matrix's data; breeze interprets a negative
  // endpoint of a Range.Inclusive as counting back from the dimension's end
  def apply(rowRange: Range, colRange: Range): DenseMatrix =
    new DenseMatrix(m(rowRange, colRange))

  def apply(rowRange: Range, allCols: ::.type): DenseMatrix =
    new DenseMatrix(m(rowRange, ::))

  def apply(allRows: ::.type, colRange: Range): DenseMatrix =
    new DenseMatrix(m(::, colRange))

  def apply(allRows: ::.type, j: Int): BDV[Double] = m(::, j)

  def apply(i: Int, allCols: ::.type): BDV[Double] = m(i, ::).t

  // unlike the Range slices, indexed selections are copies, not views
  def apply(rowIdx: IndexedSeq[Int], allCols: ::.type): DenseMatrix =
    DenseMatrix.tabulate(rowIdx.length, cols)((i, j) => m(rowIdx(i), j))

  def apply(allRows: ::.type, colIdx: IndexedSeq[Int]): DenseMatrix =
    DenseMatrix.tabulate(rows, colIdx.length)((i, j) => m(i, colIdx(j)))

  def apply(rowIdx: IndexedSeq[Int], colIdx: IndexedSeq[Int]): DenseMatrix =
    DenseMatrix.tabulate(rowIdx.length, colIdx.length)((i, j) => m(rowIdx(i), colIdx(j)))

  // row- and column-broadcasting, as breeze's `m(*, ::)` and `m(::, *)`;
  // hand-rolled because breeze's broadcasts map through the broken
  // `canMapValues`
  def apply(eachRow: breeze.linalg.*.type, allCols: ::.type): BroadcastedRows =
    new BroadcastedRows(m)

  def apply(allRows: ::.type, eachCol: breeze.linalg.*.type): BroadcastedColumns =
    new BroadcastedColumns(m)

  def t: DenseMatrix = new DenseMatrix(m.t)
  def inv: DenseMatrix = new DenseMatrix(breeze.linalg.inv(m))
  def diag: BDV[Double] = breeze.linalg.diag(m)
  def cholesky: DenseMatrix = new DenseMatrix(breeze.linalg.cholesky(m))

  // hand-rolled: breeze's horzcat copies through falsely-contiguous views
  def horzcat(that: DenseMatrix): DenseMatrix = {
    require(rows == that.rows)
    val r = zeros(rows, cols + that.cols)
    r(::, 0 until cols) := this: Unit
    r(::, cols until r.cols) := that: Unit
    r
  }

  // always a compact, column-major copy
  def map(f: Double => Double): DenseMatrix =
    new DenseMatrix(new BDM(rows, cols, mapArray(f)))

  // computes with the actual strides rather than assuming contiguity, so it
  // is correct on views and transposes
  private def mapArray(f: Double => Double): Array[Double] = {
    val (rowStride, colStride) = if (isTranspose) (m.majorStride, 1) else (1, m.majorStride)
    val a = new Array[Double](rows * cols)
    var k = 0
    var j = 0
    while (j < cols) {
      val base = m.offset + j * colStride
      var i = 0
      while (i < rows) {
        a(k) = f(data(base + i * rowStride))
        k += 1
        i += 1
      }
      j += 1
    }
    a
  }

  def :=(that: DenseMatrix): DenseMatrix = {
    require(rows == that.rows && cols == that.cols)
    // `copy` is safe: its fast path can only fire when truly contiguous
    val b = if (data eq that.data) that.m.copy else that.m
    if (m.isTranspose == b.isTranspose) {
      val (major, minor) = if (m.isTranspose) (rows, cols) else (cols, rows)
      var j = 0
      while (j < major) {
        System.arraycopy(
          b.data,
          b.offset + j * b.majorStride,
          m.data,
          m.offset + j * m.majorStride,
          minor,
        )
        j += 1
      }
    } else {
      var j = 0
      while (j < cols) {
        var i = 0
        while (i < rows) {
          m(i, j) = b(i, j)
          i += 1
        }
        j += 1
      }
    }
    this
  }

  def :=(x: Double): DenseMatrix = {
    val (major, minor) = if (m.isTranspose) (rows, cols) else (cols, rows)
    var j = 0
    while (j < major) {
      val start = m.offset + j * m.majorStride
      util.Arrays.fill(m.data, start, start + minor, x)
      j += 1
    }
    this
  }

  def unary_- : DenseMatrix = map(-_)

  def +(that: DenseMatrix): DenseMatrix = new DenseMatrix(m + that.m)
  def -(that: DenseMatrix): DenseMatrix = new DenseMatrix(m - that.m)
  def *(that: DenseMatrix): DenseMatrix = new DenseMatrix(m * that.m)
  def *:*(that: DenseMatrix): DenseMatrix = new DenseMatrix(m *:* that.m)
  def /:/(that: DenseMatrix): DenseMatrix = new DenseMatrix(m /:/ that.m)

  def +(x: Double): DenseMatrix = new DenseMatrix(m + x)
  def -(x: Double): DenseMatrix = new DenseMatrix(m - x)
  def *(x: Double): DenseMatrix = new DenseMatrix(m * x)
  def /(x: Double): DenseMatrix = new DenseMatrix(m / x)

  def *(v: BDV[Double]): BDV[Double] = m * v

  def *(v: BVector[Double]): BDV[Double] =
    v match {
      case v: BDV[_] => m * v
      case _ => m * BDV(v.toArray)
    }

  def \(v: BDV[Double]): BDV[Double] = m \ v

  override def toString: String = m.toString

  // column-major
  def toArray: Array[Double] =
    if (isCompact && !isTranspose) util.Arrays.copyOf(data, size)
    else mapArray(x => x)

  def forceSymmetry(): Unit = {
    require(rows == cols, "only square matrices can be made symmetric")

    var i = 0
    while (i < rows) {
      var j = i + 1
      while (j < rows) {
        m(i, j) = m(j, i)
        j += 1
      }
      i += 1
    }
  }

  // the data array is exactly the logical elements in order; the size check
  // alone is not enough, as a reversed full slice (negative stride) keeps the
  // backing array's length
  def isCompact: Boolean =
    m.offset == 0 &&
      m.majorStride == (if (isTranspose) cols else rows) &&
      rows.toLong * cols == data.length

  def toCompactData(forceRowMajor: Boolean = false): (Array[Double], Boolean) =
    if (isCompact && (!forceRowMajor || isTranspose)) (data, isTranspose)
    else if (forceRowMajor) (t.toArray, true)
    else (toArray, false)

  // caller must close
  def write(os: OutputStream, forceRowMajor: Boolean, bufferSpec: BufferSpec): Unit = {
    val (a, isT) = toCompactData(forceRowMajor)
    assert(a.length == rows * cols)

    val out = bufferSpec.buildOutputBuffer(os)

    out.writeInt(rows)
    out.writeInt(cols)
    out.writeBoolean(isT)
    out.writeDoubles(a)
    out.flush()
  }

  def write(fs: FS, path: String, forceRowMajor: Boolean = false, bufferSpec: BufferSpec): Unit =
    using(fs.create(path))(os => write(os, forceRowMajor, bufferSpec))
}

object DenseMatrix {

  final class BroadcastedRows(private val m: BDM[Double]) extends AnyVal {
    def +(v: BDV[Double]): DenseMatrix = map(_ + v)
    def -(v: BDV[Double]): DenseMatrix = map(_ - v)
    def *:*(v: BDV[Double]): DenseMatrix = map(_ *:* v)
    def /:/(v: BDV[Double]): DenseMatrix = map(_ /:/ v)

    def map(f: BDV[Double] => BDV[Double]): DenseMatrix = {
      val r = BDM.zeros[Double](m.rows, m.cols)
      var i = 0
      while (i < m.rows) {
        val v = f(m(i, ::).t)
        require(v.length == m.cols)
        var j = 0
        while (j < m.cols) {
          r(i, j) = v(j)
          j += 1
        }
        i += 1
      }
      new DenseMatrix(r)
    }

    def map(f: BDV[Double] => Double): BDV[Double] = {
      val a = new Array[Double](m.rows)
      var i = 0
      while (i < m.rows) {
        a(i) = f(m(i, ::).t)
        i += 1
      }
      BDV(a)
    }
  }

  final class BroadcastedColumns(private val m: BDM[Double]) extends AnyVal {
    def +(v: BDV[Double]): DenseMatrix = map(_ + v)
    def -(v: BDV[Double]): DenseMatrix = map(_ - v)
    def *:*(v: BDV[Double]): DenseMatrix = map(_ *:* v)
    def /:/(v: BDV[Double]): DenseMatrix = map(_ /:/ v)

    def map(f: BDV[Double] => BDV[Double]): DenseMatrix = {
      val r = BDM.zeros[Double](m.rows, m.cols)
      var j = 0
      while (j < m.cols) {
        val v = f(m(::, j))
        require(v.length == m.rows)
        var i = 0
        while (i < m.rows) {
          r(i, j) = v(i)
          i += 1
        }
        j += 1
      }
      new DenseMatrix(r)
    }
  }

  def apply(rows: Int, cols: Int, data: Array[Double], isTranspose: Boolean = false)
    : DenseMatrix = {
    require(rows.toLong * cols <= data.length)

    new DenseMatrix(
      new BDM(
        rows = rows,
        cols = cols,
        data = data,
        offset = 0,
        majorStride = if (isTranspose) cols else rows,
        isTranspose = isTranspose,
      )
    )
  }

  // literal rows, as breeze's `DenseMatrix((1.0, 2.0), (3.0, 4.0))`; bounding
  // R by Product keeps this overload out of non-tuple applications
  def apply[R <: Product](rows: R*)(implicit rl: LiteralRow[R, Double]): DenseMatrix =
    new DenseMatrix(BDM(rows: _*))

  def zeros(rows: Int, cols: Int): DenseMatrix =
    new DenseMatrix(BDM.zeros[Double](rows, cols))

  def fill(rows: Int, cols: Int)(v: => Double): DenseMatrix =
    new DenseMatrix(BDM.fill(rows, cols)(v))

  def ones(rows: Int, cols: Int): DenseMatrix =
    new DenseMatrix(BDM.ones[Double](rows, cols))

  def rand(rows: Int, cols: Int, dist: Rand[Double] = Rand.uniform): DenseMatrix =
    new DenseMatrix(BDM.rand(rows, cols, dist))

  def tabulate(rows: Int, cols: Int)(f: (Int, Int) => Double): DenseMatrix =
    new DenseMatrix(BDM.tabulate(rows, cols)(f))

  // a * b.t
  def outer(a: BDV[Double], b: BDV[Double]): DenseMatrix = new DenseMatrix(a * b.t)

  def lowerTriangular(x: DenseMatrix): DenseMatrix =
    new DenseMatrix(breeze.linalg.lowerTriangular(x.m))

  case class QR(q: DenseMatrix, r: DenseMatrix)

  def qrReduced(x: DenseMatrix): QR = {
    val breeze.linalg.qr.QR(q, r) = breeze.linalg.qr.reduced(x.m)
    QR(new DenseMatrix(q), new DenseMatrix(r))
  }

  def qrReducedJustQ(x: DenseMatrix): DenseMatrix =
    new DenseMatrix(breeze.linalg.qr.reduced.justQ(x.m))

  // c += a * b, with c compact and column-major
  private[linalg] def fma(c: DenseMatrix, a0: DenseMatrix, b0: DenseMatrix): Unit = {
    assert(a0.cols == b0.rows)
    assert(c.isCompact && !c.isTranspose)

    def dgemmReady(x: BDM[Double]): BDM[Double] =
      if (x.majorStride < math.max(if (x.isTranspose) x.cols else x.rows, 1)) x.copy else x

    val a = dgemmReady(a0.m)
    val b = dgemmReady(b0.m)

    import dev.ludovic.netlib.blas.BLAS.{getInstance => blas}
    blas.dgemm(
      if (a.isTranspose) "T" else "N",
      if (b.isTranspose) "T" else "N",
      c.rows,
      c.cols,
      a.cols,
      1.0,
      a.data,
      a.offset,
      a.majorStride,
      b.data,
      b.offset,
      b.majorStride,
      1.0,
      c.data,
      0,
      c.rows,
    )
  }

  // assumes data isCompact, caller must close
  def read(is: InputStream, bufferSpec: BufferSpec): DenseMatrix = {
    val in = bufferSpec.buildInputBuffer(is)

    val rows = in.readInt()
    val cols = in.readInt()
    val isTranspose = in.readBoolean()

    val data = new Array[Double](rows * cols)
    in.readDoubles(data)

    apply(rows, cols, data, isTranspose)
  }

  def read(fs: FS, path: String, bufferSpec: BufferSpec): DenseMatrix =
    using(new DataInputStream(fs.open(path)))(is => read(is, bufferSpec))

  def importFromDoubles(fs: FS, path: String, nRows: Int, nCols: Int, rowMajor: Boolean)
    : DenseMatrix = {
    require(nRows * nCols.toLong <= Int.MaxValue)
    val data = ArrayImpex.importFromDoubles(fs, path, nRows * nCols)

    apply(nRows, nCols, data, rowMajor)
  }

  def exportToDoubles(fs: FS, path: String, m: DenseMatrix, forceRowMajor: Boolean): Boolean = {
    val (data, rowMajor) = m.toCompactData(forceRowMajor)
    assert(data.length == m.rows * m.cols)

    ArrayImpex.exportToDoubles(fs, path, data)

    rowMajor
  }

  // lets the elementwise UFuncs (breeze.numerics: exp, sqrt, abs, pow, ...)
  // apply to the facade by mapping the scalar op over each element
  implicit def impl_ElementwiseUFunc_DM[Op <: ElementwiseUFunc](
    implicit op: UFunc.UImpl[Op, Double, Double]
  ): UFunc.UImpl[Op, DenseMatrix, DenseMatrix] =
    (x: DenseMatrix) => x.map(op(_))

  implicit def impl_ElementwiseUFunc_DM_S[Op <: ElementwiseUFunc](
    implicit op: UFunc.UImpl2[Op, Double, Double, Double]
  ): UFunc.UImpl2[Op, DenseMatrix, Double, DenseMatrix] =
    (x: DenseMatrix, s: Double) => x.map(op(_, s))

  // backs the reductions (sum, max, ...); breeze's traversal has its own,
  // correct stride check, so delegation is safe
  implicit val impl_CanTraverseValues_DM: CanTraverseValues[DenseMatrix, Double] =
    new CanTraverseValues[DenseMatrix, Double] {
      private def delegate = implicitly[CanTraverseValues[BDM[Double], Double]]

      override def traverse(from: DenseMatrix, fn: ValuesVisitor[Double]): fn.type =
        delegate.traverse(from.m, fn)

      override def isTraversableAgain(from: DenseMatrix): Boolean =
        delegate.isTraversableAgain(from.m)
    }

  // the lapack-backed factorizations copy their input, so delegating is safe
  implicit val impl_svd_DM: svd.Impl[DenseMatrix, svd.SVD[DenseMatrix, BDV[Double]]] =
    (x: DenseMatrix) => {
      val svd.SVD(u, s, vt) = svd(x.m)
      svd.SVD(new DenseMatrix(u), s, new DenseMatrix(vt))
    }

  implicit val impl_eigSym_DM: eigSym.Impl[DenseMatrix, eigSym.EigSym[BDV[Double], DenseMatrix]] =
    (x: DenseMatrix) => {
      val eigSym.EigSym(values, vectors) = eigSym(x.m)
      eigSym.EigSym(values, new DenseMatrix(vectors))
    }
}

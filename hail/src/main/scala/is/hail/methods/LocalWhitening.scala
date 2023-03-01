package is.hail.methods

import is.hail.annotations.Region
import is.hail.asm4s.{Value, _}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.interfaces.SNDArray.assertColMajor
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.interfaces.{ColonIndex => Colon}
import is.hail.types.physical.{PCanonicalNDArray, PFloat64Required}
import is.hail.utils.FastIndexedSeq

class LocalWhitening(cb: EmitCodeBuilder, vecSize: SizeValue, _w: Value[Long], chunksize: Value[Long], _blocksize: Value[Long], region: Value[Region], normalizeAfterWhitening: Boolean) {
  val m = vecSize
  val w = SizeValueDyn(cb.memoizeField(_w))
  val b = SizeValueDyn(cb.memoizeField(chunksize))
  val wpb = SizeValueDyn(cb.memoizeField(w+b))

  val curSize = cb.newField[Long]("curSize", 0)
  val pivot = cb.newField[Long]("pivot", 0)

  val matType = PCanonicalNDArray(PFloat64Required, 2)
  val vecType = PCanonicalNDArray(PFloat64Required, 1)

  val (tsize, worksize) = SNDArray.geqr_query(cb, m, chunksize, region)

  val Q = cb.memoizeField(matType.constructUninitialized(FastIndexedSeq(m, w), cb, region), "LW_Q").asNDArray.coerceToShape(cb, m, w)
  val R = cb.memoizeField(matType.constructUninitialized(FastIndexedSeq(w, w), cb, region), "LW_R").asNDArray.coerceToShape(cb, w, w)
  val work1 = cb.memoizeField(matType.constructUninitialized(FastIndexedSeq(wpb, wpb), cb, region), "LW_work1").asNDArray.coerceToShape(cb, wpb, wpb)
  val work2 = cb.memoizeField(matType.constructUninitialized(FastIndexedSeq(wpb, b), cb, region), "LW_work2").asNDArray.coerceToShape(cb, wpb, b)
  val Rtemp = cb.memoizeField(matType.constructUninitialized(FastIndexedSeq(wpb, wpb), cb, region), "LW_Rtemp").asNDArray.coerceToShape(cb, wpb, wpb)
  val Qtemp = cb.memoizeField(matType.constructUninitialized(FastIndexedSeq(m, b), cb, region), "LW_Qtemp").asNDArray.coerceToShape(cb, m, b)
  val Qtemp2 = cb.memoizeField(matType.constructUninitialized(FastIndexedSeq(m, w), cb, region), "LW_Qtemp2").asNDArray.coerceToShape(cb, m, w)
  val blocksize = cb.memoizeField(_blocksize.min(w))
  val work3len = SizeValueDyn(cb.memoize(worksize.max(blocksize * m.max(wpb))))
  val work3: SNDArrayValue = cb.memoizeField(vecType.constructUninitialized(FastIndexedSeq(work3len), cb, region), "LW_work3").asNDArray
  val Tlen = SizeValueDyn(cb.memoizeField(tsize.max(blocksize*wpb)))
  val T: SNDArrayValue = cb.memoizeField(vecType.constructUninitialized(FastIndexedSeq(Tlen), cb, region), "LW_T").asNDArray

  def reset(cb: EmitCodeBuilder): Unit = {
    cb.assign(curSize, 0L)
    cb.assign(pivot, 0L)
  }

  // Pre: A1 is current window, A2 is next window, [Q1 Q2] R = [A1 A2] is qr fact
  // Post: W contains locally whitened A2, Qout R[-w:, -w:] = A2[:, -w:] is qr fact
  def whitenBlockPreOrthogonalized(cb: EmitCodeBuilder,
    Q1: SNDArrayValue, Q2: SNDArrayValue, Qout: SNDArrayValue,
    R: SNDArrayValue, W: SNDArrayValue,
    work1: SNDArrayValue, work2: SNDArrayValue,
    blocksize: Value[Long]
  ): Unit = {
    SNDArray.assertMatrix(Q1, Q2, Qout, R, work1, work2)
    SNDArray.assertColMajor(cb, "whitenBlockPreOrthogonalized", Q1, Q2, Qout, R, work1, work2)

    val Seq(m, w) = Q1.shapes
    val n = Q2.shapes(1)
    val wpn = R.shapes(0)

    cb.ifx(wpn.cne(w + n), cb._fatal("whitenBase: bad dimensions"))

    Q2.assertHasShape(cb, FastIndexedSeq(m, n), "")
    Qout.assertHasShape(cb, FastIndexedSeq(m, w), "")
    assert(R.hasShapeStatic(wpn, wpn))
    W.assertHasShape(cb, FastIndexedSeq(m, n), "")
    work1.assertHasShape(cb, FastIndexedSeq(wpn, wpn), "")
    work2.assertHasShape(cb, FastIndexedSeq(wpn, n), "")

    val i = cb.mb.newLocal[Long]("whiten_base_i")

    // set work1 to I
    work1.setToZero(cb)
    cb.forLoop(cb.assign(i, 0L), i.toL < w + n, cb.assign(i, i+1), work1.setElement(FastIndexedSeq(i, i), primitive(1.0), cb))

    cb.forLoop(cb.assign(i, 0L), i < n, cb.assign(i, i+1), {
      // Loop invariant:
      // * ([Q1 Q2] work1[:, i:w+n]) R[i:w+n, i:w+n] = [A1 A2][i:w+n] is qr fact
      // * ([Q1 Q2] work2[:, 0:i]) is locally whitened A2[:, 0:i]

      // work2[:, i] = work1[:, w+i] * R[w+i, w+i]
      val wpi = cb.newLocal[Long]("w_plus_i", w+i)
      val w1col = work1.slice(cb, Colon, wpi)
      val w2col = work2.slice(cb, Colon, i)
      SNDArray.copyVector(cb, w1col, w2col)
      if (!normalizeAfterWhitening) {
        SNDArray.scale(cb, R.loadElement(FastIndexedSeq(wpi, wpi), cb), w2col)
      }

      // work3 > blocksize * (w+n - i+1) < blocksize * (w+n)
      SNDArray.tpqrt(R.slice(cb, (i+1, null), (i+1, null)), R.slice(cb, (i, i+1), (i+1, null)), T, work3, blocksize, cb)
      SNDArray.tpmqrt("R", "N", R.slice(cb, (i, i+1), (i+1, null)), T, work1.slice(cb, Colon, (i+1, null)), work1.slice(cb, Colon, (i, i+1)), work3, blocksize, cb)
    })

    // W = [Q1 Q2] work2 is locally whitened A2
    SNDArray.gemm(cb, "N", "N", 1.0, Q1, work2.slice(cb, (null, w), Colon), 0.0, W)
    SNDArray.gemm(cb, "N", "N", 1.0, Q2, work2.slice(cb, (w, null), Colon), 1.0, W)

    // Qout = [Q1 Q2] work1, Qout R[n:w+n, n:w+n] = A2[:, n-w:n] is qr fact
    SNDArray.gemm(cb, "N", "N", 1.0, Q1, work1.slice(cb, (null, w), (n, null)), 0.0, Qout)
    SNDArray.gemm(cb, "N", "N", 1.0, Q2, work1.slice(cb, (w, null), (n, null)), 1.0, Qout)
  }

  // Cyclically permute the columns in a QR factorization.
  //
  // Pre: Let Q1 = Q[:, 0:p0], Q2 = Q[:, p0:n], R11 = R[0:p0, 0:p0], R12 = R[0:p0, p0:n], etc.
  // * [Q2 Q1] [R22 R21; 0 R11] = [A2 A1] is a qr fact
  // Post: Same, with p1 substituted for p0
  def qrPivot(cb: EmitCodeBuilder,
    Q: SNDArrayValue, R: SNDArrayValue,
    p0: Value[Long], p1: Value[Long]
  ): Unit = {
    val Seq(m, w) = Q.shapes
    val Seq(t) = T.shapes
    cb.ifx(R.shapes(0).cne(w), cb._fatal("qr_pivot: R nrows != w"))
    cb.ifx(R.shapes(1).cne(w), cb._fatal("qr_pivot: R ncols != w"))
    cb.ifx(m <= w, cb._fatal("qr_pivot: m <= w, m=", m.toS, ", w=", w.toS))
    cb.ifx(p0 < 0 || p0 >= p1 || p1 > w, cb._fatal("qr_pivot: bad p0, p1"))
    cb.ifx(t < blocksize * p0.max((p1-p0).max(w-p1)), cb._fatal("qr_pivot: T too small"))

    val r0 = (null, p0)
    val r1 = (p0, p1)
    val r2 = (p1, null)
    val r01 = (null, p1)

    val b0 = cb.memoize(blocksize.min(p0))
    val b1 = cb.memoize(blocksize.min(p1-p0))
    val b2 = cb.memoize(blocksize.min(w-p1))

    // Set lower trapezoid of R[r12, r1] to zero
    val j = cb.mb.newLocal[Long]("j")
    cb.forLoop(cb.assign(j, p0), j < p1, cb.assign(j, j+1), {
      R.slice(cb, (j+1, null), j).setToZero(cb)
    })

    R.slice(cb, r0, r1).setToZero(cb)

    cb.ifx(p1 < w, {
      SNDArray.tpqrt(R.slice(cb, r2, r2), R.slice(cb, r1, r2), T, work3, b2, cb)
      SNDArray.tpmqrt("L", "T", R.slice(cb, r1, r2), T, R.slice(cb, r2, r01), R.slice(cb, r1, r01), work3, b2, cb)
      SNDArray.tpmqrt("R", "N", R.slice(cb, r1, r2), T, Q.slice(cb, Colon, r2), Q.slice(cb, Colon, r1), work3, b2, cb)
    })
    cb.ifx(p0 > 0, {
      SNDArray.tpqrt(R.slice(cb, r0, r0), R.slice(cb, r1, r0), T, work3, b0, cb)
      SNDArray.tpmqrt("L", "T", R.slice(cb, r1, r0), T, R.slice(cb, r0, r1), R.slice(cb, r1, r1), work3, b0, cb)
      SNDArray.tpmqrt("R", "N", R.slice(cb, r1, r0), T, Q.slice(cb, Colon, r0), Q.slice(cb, Colon, r1), work3, b0, cb)
    })
    SNDArray.geqrt(R.slice(cb, r1, r1), T, work3, b1, cb)
    SNDArray.gemqrt("R", "N", R.slice(cb, r1, r1), T, Q.slice(cb, Colon, r1), work3, b1, cb)
  }

  // Whiten block A, where A is no smaller than the window, by orthogonalizing
  // against *all* of the previous window, and then "undoing" as needed to get
  // the correct windowed orthogonalization.
  //
  // Pre: Q R = A0 is qr fact of current window, A contains next window
  // Post: A contains A_orig whitened, Q R = A_orig
  def whitenBlockSmallWindow(cb: EmitCodeBuilder,
    Q: SNDArrayValue, R: SNDArrayValue, A: SNDArrayValue,
    Qtemp: SNDArrayValue, Qtemp2: SNDArrayValue, Rtemp: SNDArrayValue,
    work1: SNDArrayValue, work2: SNDArrayValue,
    blocksize: Value[Long]
  ): Unit = {
    val Seq(m, w) = Q.shapes
    val n = A.shapes(1)
    val wpn = work1.shapes(0)

    cb.ifx(wpn.cne(w + n), cb._fatal("whitenNonrecur: bad dimensions"))

    assert(Q.hasShapeStatic(m, w))
    R.assertHasShape(cb, FastIndexedSeq(w, w), "")
    assert(A.hasShapeStatic(m, n))
    Qtemp.assertHasShape(cb, FastIndexedSeq(m, n), "")
    Qtemp2.assertHasShape(cb, FastIndexedSeq(m, w), "")
    Rtemp.assertHasShape(cb, FastIndexedSeq(wpn, wpn), "")
    work1.assertHasShape(cb, FastIndexedSeq(wpn, wpn), "")
    work2.assertHasShape(cb, FastIndexedSeq(wpn, n), "")

    val r0 = (null, w)
    val r1 = (w, null)

    // copy upper triangle of R to Rtemp[r0, r0]
    SNDArray.copyMatrix(cb, "U", R, Rtemp.slice(cb, r0, r0))

    // orthogonalize against Q
    // Rtemp[r0, r1] = Q' A
    SNDArray.gemm(cb, "T", "N", Q, A, Rtemp.slice(cb, r0, r1))
    // A = A - Q Rtemp[r0, r1]
    SNDArray.gemm(cb, "N", "N", -1.0, Q, Rtemp.slice(cb, r0, r1), 1.0, A)

    // Compute QR fact of A; store R fact in Rtemp[r1, r1], Q fact in Qtemp
    // work3 > geqr_query(m, n)
    SNDArray.geqr_full(cb, A, Qtemp, Rtemp.slice(cb, r1, r1), T, work3)

    // now Qtemp Rtemp[r1, r1] = A_orig - Q Rtemp[r0, r1]
    // so Q Rtemp[r0, r1] + Qtemp Rtemp[r1, r1] = A_orig
    // and [Q Qtemp] R = [A0 A_orig]
    whitenBlockPreOrthogonalized(cb, Q, Qtemp, Qtemp2, Rtemp, A, work1, work2, blocksize)

    // copy upper triangle of Rtemp[n:w+n, n:w+n] to R
    SNDArray.copyMatrix(cb, "U", Rtemp.slice(cb, (n, w+n), (n, w+n)), R)
    // copy Qtemp2 to Q
    SNDArray.copyMatrix(cb, " ", Qtemp2, Q)
    // now Q R = A_orig[::, n-w:n]
  }

  // Whiten block A, where A is no larger than the window, by orthogonalizing
  // all of A against the "newest" cols of Q, those which are within the window
  // of all cols of A. Then whiten A against the "oldest" cols of Q, with the
  // smaller effective window size.
  //
  // Pre: Let b = A.shapes(1), Q1 = Q[:, 0:p], Q2 = Q[:, p:p+b], Q3 = Q[:, p+b:w]
  // * [Q2 Q3 Q1] [R22 R23 R21; 0 R33 R31; 0 0 R11] = [A2 A3 A1] is a qr fact
  // Post:
  // * [Q3 Q1 Q2] [R33 R31 R32; 0 R11 R12; 0 0 R22] = [A3 A1 A_orig] is a qr fact
  // * A contains whitened A_orig
  def whitenBlockLargeWindow(cb: EmitCodeBuilder,
    Q: SNDArrayValue, R: SNDArrayValue, p: Value[Long], A: SNDArrayValue, Qtemp: SNDArrayValue,
    Qtemp2: SNDArrayValue, Rtemp: SNDArrayValue, work1: SNDArrayValue, work2: SNDArrayValue,
    blocksize: Value[Long]
  ): Unit = {
    val b = A.shapes(1)
    val bb = Rtemp.shapes(0)

    cb.ifx((b*2).cne(bb), cb._fatal("whitenStep: invalid dimensions"))

    assert(Q.hasShapeStatic(m, w))
    assert(R.hasShapeStatic(w, w))
    assert(A.hasShapeStatic(m, b))
    assert(Qtemp.hasShapeStatic(m, b))
    assert(Qtemp2.hasShapeStatic(m, b))
    assert(Rtemp.hasShapeStatic(bb, bb))
    assert(work1.hasShapeStatic(bb, bb))
    assert(work2.hasShapeStatic(bb, b))

    val ppb = cb.memoize(p+b)
    qrPivot(cb, Q, R, p, ppb)
    // now [Q3 Q1 Q2] [R33 R31 R32; 0 R11 R12; 0 0 R22] = [A3 A1 A2]

    val r1 = (null, p)
    val r2 = (p, ppb)
    val r3 = (ppb, null)

    val R12 = R.slice(cb, r1, r2)
    val R22 = R.slice(cb, r2, r2)
    val R32 = R.slice(cb, r3, r2)
    val Q1 = Q.slice(cb, Colon, r1)
    val Q2 = Q.slice(cb, Colon, r2)
    val Q3 = Q.slice(cb, Colon, r3)

    // Orthogonalize against Q3
    SNDArray.gemm(cb, "T", "N", Q3, A, R32)
    SNDArray.gemm(cb, "N", "N", -1.0, Q3, R32, 1.0, A)
    // Orthogonalize against Q1
    SNDArray.gemm(cb, "T", "N", Q1, A, R12)
    SNDArray.gemm(cb, "N", "N", -1.0, Q1, R12, 1.0, A)

    // Now A = A_orig - Q3 R32 - Q1 R12
    whitenBlockSmallWindow(cb, Q2, R22, A, Qtemp, Qtemp2, Rtemp, work1, work2, blocksize)
    // now A contains A_orig - Q3 R32 - Q1 R12 whitened against A2
    // and Q2 R22 = A = A_orig - Q3 R32 - Q1 R12
    // so A_orig = Q3 R32 + Q1 R12 + Q2 R22
  }

  def whitenBlock(cb: EmitCodeBuilder, _A: SNDArrayValue): Unit = {
    assertColMajor(cb, "whitenBlock", _A)
    val b = _A.shapes(1)

    val A = _A.coerceToShape(cb, m, b)
    cb.ifx(b > chunksize, cb._fatal("whitenBlock: A too large, found ", b.toS, ", expected ", chunksize.toS))

    cb.ifx(curSize < w, {
      // Orthogonalize against existing Q
      val Rslice = R.slice(cb, (null, curSize), (curSize, curSize + b))
      val Qslice = Q.slice(cb, Colon, (null, curSize))
      // Rslice = Q' A
      SNDArray.gemm(cb, "T", "N", Qslice, A, Rslice)
      // A = A - Q Rslice
      SNDArray.gemm(cb, "N", "N", -1.0, Qslice, Rslice, 1.0, A)

      // Compute QR fact of A; store R fact in Rtemp[r1, r1], Q fact in Qtemp
      val Rslice2 = R.slice(cb, (curSize, curSize + b), (curSize, curSize + b))
      val Qslice2 = Q.slice(cb, Colon, (curSize, curSize + b))
      SNDArray.geqr_full(cb, A, Qslice2, Rslice2, T, work3)

      // Copy whitened A back to A
      val j = cb.newLocal[Long]("j")
      cb.forLoop(cb.assign(j, 0L), j < b, cb.assign(j, j+1), {
        val Acol = A.slice(cb, Colon, j)
        SNDArray.copyVector(cb, Qslice2.slice(cb, Colon, j), Acol)
        SNDArray.scale(cb, Rslice2.loadElement(FastIndexedSeq(j, j), cb), Acol)
      })

      cb.assign(curSize, curSize + b)
    }, {
      cb.ifx(curSize.cne(w), cb._fatal("whitenBlock: initial blocks didn't evenly divide window size"))

      val bb = SizeValueDyn(cb.memoize(b*2))
      whitenBlockLargeWindow(cb,
        Q, R, pivot, A,
        Qtemp.slice(cb, Colon, (null, b)),
        Qtemp2.slice(cb, Colon, (null, b)),
        Rtemp.slice(cb, (null, bb), (null, bb)),
        work1.slice(cb, (null, bb), (null, bb)),
        work2.slice(cb, (null, bb), (null, b)),
        cb.memoize(blocksize.min(b)))

      cb.assign(pivot, pivot + b)
      cb.ifx(pivot >= w, {
        cb.ifx(pivot.cne(w), cb._fatal("whitenBlock, blocks didn't evenly divide window size"))
        cb.assign(pivot, 0L)
      })
    })
  }

  def initializeWindow(cb: EmitCodeBuilder, _A: SNDArrayValue): Unit = {
    val b = _A.shapes(1)

    val A = _A.coerceToShape(cb, m, b)
    cb.ifx(b > w, cb._fatal("initializeWindow: A too large"))
    cb.ifx(curSize.cne(0), cb._fatal("initializeWindow: can only be called on empty state"))

    val Rslice = R.slice(cb, (null, b), (null, b))
    val Qslice = Q.slice(cb, Colon, (null, b))
    SNDArray.geqr_full(cb, A, Qslice, Rslice, T, work3)
    cb.assign(curSize, b)
  }
}

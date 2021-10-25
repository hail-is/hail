package is.hail.types.physical.stypes.interfaces

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.{RNDArray, TypeWithRequiredness}
import is.hail.types.physical.{PCanonicalNDArray, PFloat64Required, PNDArray, PNumeric, PPrimitive, PType}
import is.hail.types.physical.stypes.concrete.{SNDArraySlice, SNDArraySliceCode, SNDArraySliceValue}
import is.hail.linalg.{BLAS, LAPACK}
import is.hail.types.physical.stypes.concrete.{SNDArraySlice, SNDArraySliceCode}
import is.hail.types.physical.stypes.primitives.SFloat64Code
import is.hail.types.physical.stypes.{EmitType, SCode, SSettable, SType, SValue}
import is.hail.utils.{FastIndexedSeq, toRichIterable, valueToRichCodeRegion}

import scala.collection.mutable

object SNDArray {
  def numElements(shape: IndexedSeq[Value[Long]]): Code[Long] = {
    shape.foldLeft(1L: Code[Long])(_ * _)
  }

  // Column major order
  def forEachIndexColMajor(cb: EmitCodeBuilder, shape: IndexedSeq[Value[Long]], context: String)
                          (f: (EmitCodeBuilder, IndexedSeq[Value[Long]]) => Unit): Unit = {
    forEachIndexWithInitAndIncColMajor(cb, shape, shape.map(_ => (cb: EmitCodeBuilder) => ()), shape.map(_ => (cb: EmitCodeBuilder) => ()), context)(f)
  }

  def coiterate(cb: EmitCodeBuilder, arrays: (SNDArrayValue, String)*)(body: IndexedSeq[SValue] => Unit): Unit = {
    if (arrays.isEmpty) return
    val indexVars = Array.tabulate(arrays(0)._1.st.nDims)(i => s"i$i").toFastIndexedSeq
    val indices = Array.range(0, arrays(0)._1.st.nDims).toFastIndexedSeq
    coiterate(cb, indexVars, arrays.map { case (array, name) => (array, indices, name) }: _*)(body)
  }

  // Note: to iterate through an array in column major order, make sure the indices are in ascending order. E.g.
  // A.coiterate(cb, IndexedSeq("i", "j"), (A, IndexedSeq(0, 1), "A"), (B, IndexedSeq(0, 1), "B"), {
  //   SCode.add(cb, a, b)
  // })
  // computes A += B.
  def coiterate(
    cb: EmitCodeBuilder,
    indexVars: IndexedSeq[String],
    arrays: (SNDArrayValue, IndexedSeq[Int], String)*
  )(body: IndexedSeq[SValue] => Unit
  ): Unit = {
    _coiterate(cb, indexVars, arrays: _*) { ptrs =>
      val codes = ptrs.zip(arrays).map { case (ptr, (array, _, _)) =>
        val pt = array.st.pType.elementType
        pt.loadCheapSCode(cb, pt.loadFromNested(ptr))
      }
      body(codes)
    }
  }

  def _coiterate(
    cb: EmitCodeBuilder,
    indexVars: IndexedSeq[String],
    arrays: (SNDArrayValue, IndexedSeq[Int], String)*
  )(body: IndexedSeq[Value[Long]] => Unit
  ): Unit = {
    val indexSizes = new Array[Settable[Int]](indexVars.length)
    val indexCoords = Array.tabulate(indexVars.length) { i => cb.newLocal[Int](indexVars(i)) }

    case class ArrayInfo(
      array: SNDArrayValue,
      strides: IndexedSeq[Value[Long]],
      pos: IndexedSeq[Settable[Long]],
      indexToDim: Map[Int, Int],
      name: String)

    val info = arrays.toIndexedSeq.map { case (array, indices, name) =>
      for (idx <- indices) assert(idx < indexVars.length && idx >= 0)
      // FIXME: relax this assumption to handle transposing, non-column major
      for (i <- 0 until indices.length - 1) assert(indices(i) < indices(i+1))
      assert(indices.length == array.st.nDims)

      val shape = array.shapes
      for (i <- indices.indices) {
        val idx = indices(i)
        if (indexSizes(idx) == null) {
          indexSizes(idx) = cb.newLocal[Int](s"${indexVars(idx)}_max")
          cb.assign(indexSizes(idx), shape(i).toI)
        } else {
          cb.ifx(indexSizes(idx).cne(shape(i).toI), cb._fatal(s"${indexVars(idx)} indexes incompatible dimensions"))
        }
      }
      val strides = array.strides
      val pos = Array.tabulate(array.st.nDims + 1) { i => cb.newLocal[Long](s"$name$i") }
      val indexToDim = indices.zipWithIndex.toMap
      ArrayInfo(array, strides, pos, indexToDim, name)
    }

    def recurLoopBuilder(idx: Int): Unit = {
      if (idx < 0) {
        // FIXME: to handle non-column major, need to use `pos` of smallest index var
        body(info.map(_.pos(0)))
      } else {
        val coord = indexCoords(idx)
        def init(): Unit = {
          cb.assign(coord, 0)
          for (n <- arrays.indices) {
            if (info(n).indexToDim.contains(idx)) {
              val i = info(n).indexToDim(idx)
              // FIXME: assumes array's indices in ascending order
              cb.assign(info(n).pos(i), info(n).pos(i+1))
            }
          }
        }
        def increment(): Unit = {
          cb.assign(coord, coord + 1)
          for (n <- arrays.indices) {
            if (info(n).indexToDim.contains(idx)) {
              val i = info(n).indexToDim(idx)
              cb.assign(info(n).pos(i), info(n).pos(i) + info(n).strides(i))
            }
          }
        }

        cb.forLoop(init(), coord < indexSizes(idx), increment(), recurLoopBuilder(idx - 1))
      }
    }

    for (n <- arrays.indices) {
      cb.assign(info(n).pos(info(n).array.st.nDims), info(n).array.firstDataAddress)
    }
    recurLoopBuilder(indexVars.length - 1)
  }

  // Column major order
  def forEachIndexWithInitAndIncColMajor(cb: EmitCodeBuilder, shape: IndexedSeq[Value[Long]], inits: IndexedSeq[EmitCodeBuilder => Unit],
                                         incrementers: IndexedSeq[EmitCodeBuilder => Unit], context: String)
                                        (f: (EmitCodeBuilder, IndexedSeq[Value[Long]]) => Unit): Unit = {

    val indices = Array.tabulate(shape.length) { dimIdx => cb.newLocal[Long](s"${ context }_foreach_dim_$dimIdx", 0L) }

    def recurLoopBuilder(dimIdx: Int, innerLambda: () => Unit): Unit = {
      if (dimIdx == shape.length) {
        innerLambda()
      }
      else {
        val dimVar = indices(dimIdx)

        recurLoopBuilder(dimIdx + 1,
          () => {
            cb.forLoop({
              inits(dimIdx)(cb)
              cb.assign(dimVar, 0L)
            }, dimVar < shape(dimIdx), {
              incrementers(dimIdx)(cb)
              cb.assign(dimVar, dimVar + 1L)
            },
              innerLambda()
            )
          }
        )
      }
    }

    val body = () => f(cb, indices)

    recurLoopBuilder(0, body)
  }

  // Row major order
  def forEachIndexRowMajor(cb: EmitCodeBuilder, shape: IndexedSeq[Value[Long]], context: String)
                          (f: (EmitCodeBuilder, IndexedSeq[Value[Long]]) => Unit): Unit = {
    forEachIndexWithInitAndIncRowMajor(cb, shape, shape.map(_ => (cb: EmitCodeBuilder) => ()), shape.map(_ => (cb: EmitCodeBuilder) => ()), context)(f)
  }

  // Row major order
  def forEachIndexWithInitAndIncRowMajor(cb: EmitCodeBuilder, shape: IndexedSeq[Value[Long]], inits: IndexedSeq[EmitCodeBuilder => Unit],
                                         incrementers: IndexedSeq[EmitCodeBuilder => Unit], context: String)
                                        (f: (EmitCodeBuilder, IndexedSeq[Value[Long]]) => Unit): Unit = {

    val indices = Array.tabulate(shape.length) { dimIdx => cb.newLocal[Long](s"${ context }_foreach_dim_$dimIdx", 0L) }

    def recurLoopBuilder(dimIdx: Int, innerLambda: () => Unit): Unit = {
      if (dimIdx == -1) {
        innerLambda()
      }
      else {
        val dimVar = indices(dimIdx)

        recurLoopBuilder(dimIdx - 1,
          () => {
            cb.forLoop({
              inits(dimIdx)(cb)
              cb.assign(dimVar, 0L)
            }, dimVar < shape(dimIdx), {
              incrementers(dimIdx)(cb)
              cb.assign(dimVar, dimVar + 1L)
            },
              innerLambda()
            )
          }
        )
      }
    }

    val body = () => f(cb, indices)

    recurLoopBuilder(shape.length - 1, body)
  }

  // Column major order
  def unstagedForEachIndex(shape: IndexedSeq[Long])
                          (f: IndexedSeq[Long] => Unit): Unit = {

    val indices = Array.tabulate(shape.length) {dimIdx =>  0L}

    def recurLoopBuilder(dimIdx: Int, innerLambda: () => Unit): Unit = {
      if (dimIdx == shape.length) {
        innerLambda()
      }
      else {

        recurLoopBuilder(dimIdx + 1,
          () => {
            (0 until shape(dimIdx).toInt).foreach(_ => {
              innerLambda()
              indices(dimIdx) += 1
            })
          }
        )
      }
    }

    val body = () => f(indices)

    recurLoopBuilder(0, body)
  }

  def assertMatrix(nds: SNDArrayValue*): Unit = {
    for (nd <- nds) assert(nd.st.nDims == 2)
  }

  def assertVector(nds: SNDArrayValue*): Unit = {
    for (nd <- nds) assert(nd.st.nDims == 1)
  }

  def assertColMajor(cb: EmitCodeBuilder, nds: SNDArrayValue*): Unit = {
    for (nd <- nds) {
      cb.ifx(nd.strides(0).cne(nd.st.pType.elementType.byteSize),
        cb._fatal("Require column major: found row stride ", nd.strides(0).toS, ", expected ", nd.st.pType.elementType.byteSize.toString))
    }
  }

  def copyVector(cb: EmitCodeBuilder, X: SNDArrayValue, Y: SNDArrayValue): Unit = {
    val Seq(n) = X.shapes

    Y.assertHasShape(cb, FastIndexedSeq(n), "copy: vectors have different sizes: ", Y.shapes(0).toS, ", ", n.toS)
    val ldX = X.eltStride(0).max(1)
    val ldY = Y.eltStride(0).max(1)
    cb += Code.invokeScalaObject5[Int, Long, Int, Long, Int, Unit](BLAS.getClass, "dcopy",
      n.toI,
      X.firstDataAddress, ldX,
      Y.firstDataAddress, ldY)
  }

  def copyMatrix(cb: EmitCodeBuilder, uplo: String, X: SNDArrayValue, Y: SNDArrayValue): Unit = {
    val Seq(m, n) = X.shapes
    Y.assertHasShape(cb, FastIndexedSeq(m, n), "copyMatrix: matrices have different shapes")
    val ldX = X.eltStride(1).max(1)
    val ldY = Y.eltStride(1).max(1)
    cb += Code.invokeScalaObject7[String, Int, Int, Long, Int, Long, Int, Unit](LAPACK.getClass, "dlacpy",
      uplo, m.toI, n.toI,
      X.firstDataAddress, ldX,
      Y.firstDataAddress, ldY)
  }

  def scale(cb: EmitCodeBuilder, alpha: SValue, X: SNDArrayValue): Unit =
    scale(cb, alpha.asFloat64.doubleCode(cb), X)

  def scale(cb: EmitCodeBuilder, alpha: Value[Double], X: SNDArrayValue): Unit = {
    val Seq(n) = X.shapes
    val ldX = X.eltStride(0).max(1)
    cb += Code.invokeScalaObject4[Int, Double, Long, Int, Unit](BLAS.getClass, "dscal",
      n.toI, alpha, X.firstDataAddress, ldX)
  }

  def gemv(cb: EmitCodeBuilder, trans: String, A: SNDArrayValue, X: SNDArrayValue, Y: SNDArrayValue): Unit = {
    gemv(cb, trans, 1.0, A, X, 1.0, Y)
  }

  def gemv(cb: EmitCodeBuilder, trans: String, alpha: Value[Double], A: SNDArrayValue, X: SNDArrayValue, beta: Value[Double], Y: SNDArrayValue): Unit = {
    assertMatrix(A)
    val Seq(m, n) = A.shapes
    val errMsg = "gemv: incompatible dimensions"
    if (trans == "N") {
      X.assertHasShape(cb, FastIndexedSeq(n), errMsg)
      Y.assertHasShape(cb, FastIndexedSeq(m), errMsg)
    } else {
      X.assertHasShape(cb, FastIndexedSeq(m), errMsg)
      Y.assertHasShape(cb, FastIndexedSeq(n), errMsg)
    }
    assertColMajor(cb, A)

    val ldA = A.eltStride(1).max(1)
    val ldX = X.eltStride(0).max(1)
    val ldY = Y.eltStride(0).max(1)
    cb += Code.invokeScalaObject11[String, Int, Int, Double, Long, Int, Long, Int, Double, Long, Int, Unit](BLAS.getClass, "dgemv",
      trans, m.toI, n.toI,
      alpha,
      A.firstDataAddress, ldA,
      X.firstDataAddress, ldX,
      beta,
      Y.firstDataAddress, ldY)
  }

  def gemm(cb: EmitCodeBuilder, tA: String, tB: String, A: SNDArrayValue, B: SNDArrayValue, C: SNDArrayValue): Unit =
    gemm(cb, tA, tB, 1.0, A, B, 0.0, C)

  def gemm(cb: EmitCodeBuilder, tA: String, tB: String, alpha: Value[Double], A: SNDArrayValue, B: SNDArrayValue, beta: Value[Double], C: SNDArrayValue): Unit = {
    assertMatrix(A, B, C)
    val Seq(m, n) = C.shapes
    val k = if (tA == "N") A.shapes(1) else A.shapes(0)
    val errMsg = "gemm: incompatible matrix dimensions"

    if (tA == "N")
      A.assertHasShape(cb, FastIndexedSeq(m, k), errMsg)
    else
      A.assertHasShape(cb, FastIndexedSeq(k, m), errMsg)
    if (tB == "N")
      B.assertHasShape(cb, FastIndexedSeq(k, n), errMsg)
    else
      B.assertHasShape(cb, FastIndexedSeq(n, k), errMsg)
    assertColMajor(cb, A, B, C)

    val ldA = A.eltStride(1).max(1)
    val ldB = B.eltStride(1).max(1)
    val ldC = C.eltStride(1).max(1)
    cb += Code.invokeScalaObject13[String, String, Int, Int, Int, Double, Long, Int, Long, Int, Double, Long, Int, Unit](BLAS.getClass, "dgemm",
      tA, tB, m.toI, n.toI, k.toI,
      alpha,
      A.firstDataAddress, ldA,
      B.firstDataAddress, ldB,
      beta,
      C.firstDataAddress, ldC)
  }

  def trmm(cb: EmitCodeBuilder, side: String, uplo: String, transA: String, diag: String,
    alpha: Value[Double], A: SNDArrayValue, B: SNDArrayValue): Unit = {
    assertMatrix(A, B)
    assertColMajor(cb, A, B)

    val Seq(m, n) = B.shapes
    val Seq(a0, a1) = A.shapes
    cb.ifx(a1.cne(if (side == "left") m else n), cb._fatal("trmm: incompatible matrix dimensions"))
    // Elide check in the common case that we statically know A is square
    if (a0 != a1) cb.ifx(a0 < a1, cb._fatal("trmm: A has fewer rows than cols: ", a0.toS, ", ", a1.toS))

    val ldA = A.eltStride(1).max(1)
    val ldB = B.eltStride(1).max(1)
    cb += Code.invokeScalaObject11[String, String, String, String, Int, Int, Double, Long, Int, Long, Int, Unit](BLAS.getClass, "dtrmm",
      side, uplo, transA, diag,
      m.toI, n.toI,
      alpha,
      A.firstDataAddress, ldA,
      B.firstDataAddress, ldB)
  }

  def geqrt(A: SNDArrayValue, T: SNDArrayValue, work: SNDArrayValue, blocksize: Value[Long], cb: EmitCodeBuilder): Unit = {
    if (A.st.nDims == 2) assertColMajor(cb, A) else assertVector(A)
    assertVector(work, T)

    val Seq(m, n) = if (A.st.nDims == 2) A.shapes else FastIndexedSeq(A.shapes(0), SizeValueStatic(1))
    val nb = blocksize
    val min = cb.memoize(m.min(n))
    cb.ifx((nb > min && min > 0) || nb < 1, cb._fatal("geqrt: invalid block size: ", nb.toS))
    cb.ifx(T.shapes(0) < nb*(m.min(n)), cb._fatal("geqrt: T too small"))
    cb.ifx(work.shapes(0) < nb * n, cb._fatal("geqrt: work array too small"))

    val error = cb.mb.newLocal[Int]()
    val ldA = if (A.st.nDims == 2) A.eltStride(1).max(1) else m.toI
    cb.assign(error, Code.invokeScalaObject8[Int, Int, Int, Long, Int, Long, Int, Long, Int](LAPACK.getClass, "dgeqrt",
      m.toI, n.toI, nb.toI,
      A.firstDataAddress, ldA,
      T.firstDataAddress, nb.toI.max(1),
      work.firstDataAddress))
    cb.ifx(error.cne(0), cb._fatal("LAPACK error dtpqrt. Error code = ", error.toS))
  }

  def gemqrt(side: String, trans: String, V: SNDArrayValue, T: SNDArrayValue, C: SNDArrayValue, work: SNDArrayValue, blocksize: Value[Long], cb: EmitCodeBuilder): Unit = {
    assertMatrix(V)
    assertColMajor(cb, V)
    if (C.st.nDims == 2) assertColMajor(cb, C) else assertVector(C)
    assertVector(work, T)

    assert(side == "L" || side == "R")
    assert(trans == "T" || trans == "N")
    val Seq(l, k) = V.shapes
    val Seq(m, n) = if (C.st.nDims == 2) C.shapes else FastIndexedSeq(C.shapes(0), SizeValueStatic(1))
    val nb = blocksize
    cb.ifx((nb > k && k > 0) || nb < 1, cb._fatal("gemqrt: invalid block size: ", nb.toS))
    cb.ifx(T.shapes(0) < nb*k, cb._fatal("gemqrt: invalid T size"))
    if (side == "L") {
      cb.ifx(l.cne(m), cb._fatal("gemqrt: invalid dimensions"))
      cb.ifx(work.shapes(0) < nb * n, cb._fatal("work array too small"))
    } else {
      cb.ifx(l.cne(n), cb._fatal("gemqrt: invalid dimensions"))
      cb.ifx(work.shapes(0) < nb * m, cb._fatal("work array too small"))
    }

    val error = cb.mb.newLocal[Int]()
    val ldV = V.eltStride(1).max(1)
    val ldC = if (C.st.nDims == 2) C.eltStride(1).max(1) else m.toI
    cb.assign(error, Code.invokeScalaObject13[String, String, Int, Int, Int, Int, Long, Int, Long, Int, Long, Int, Long, Int](LAPACK.getClass, "dgemqrt",
      side, trans, m.toI, n.toI, k.toI, nb.toI,
      V.firstDataAddress, ldV,
      T.firstDataAddress, nb.toI.max(1),
      C.firstDataAddress, ldC,
      work.firstDataAddress))
    cb.ifx(error.cne(0), cb._fatal("LAPACK error dgemqrt. Error code = ", error.toS))
  }

  // Computes the QR factorization of A. Stores resulting factors in Q and R, overwriting A.
  def geqrt_full(cb: EmitCodeBuilder, A: SNDArrayValue, Q: SNDArrayValue, R: SNDArrayValue, T: SNDArrayValue, work: SNDArrayValue, blocksize: Value[Long]): Unit = {
    val Seq(m, n) = A.shapes
    SNDArray.geqrt(A, T, work, blocksize, cb)
    // copy upper triangle of A0 to R
    SNDArray.copyMatrix(cb, "U", A.slice(cb, (null, n), ::), R)

    // Set Q to I
    Q.setToZero(cb)
    val i = cb.mb.newLocal[Long]("i")
    cb.forLoop(cb.assign(i, 0L), i < n, cb.assign(i, i+1), {
      Q.setElement(FastIndexedSeq(i, i), primitive(1.0), cb)
    })
    SNDArray.gemqrt("L", "N", A, T, Q, work, blocksize, cb)
  }

  def geqr_query(cb: EmitCodeBuilder, m: Value[Long], n: Value[Long], region: Value[Region]): (Value[Long], Value[Long]) = {
    val T = cb.memoize(region.allocate(8L * 5, 8L))
    val work = cb.memoize(region.allocate(8L, 8L))
    val info = cb.memoize(Code.invokeScalaObject8[Int, Int, Long, Int, Long, Int, Long, Int, Int](LAPACK.getClass, "dgeqr",
      m.toI, n.toI,
      0, m.toI,
      T, -1,
      work, -1))
    cb.ifx(info.cne(0), cb._fatal(s"LAPACK error DGEQR. Failed size query. Error code = ", info.toS))
    val Tsize = cb.memoize(Region.loadDouble(T).toL)
    val LWork = cb.memoize(Region.loadDouble(work).toL)
    (cb.memoize(Tsize.max(5)), cb.memoize(LWork.max(1)))
  }

  def geqr(cb: EmitCodeBuilder, A: SNDArrayValue, T: SNDArrayValue, work: SNDArrayValue): Unit = {
    assertMatrix(A)
    assertColMajor(cb, A)
    assertVector(T, work)

    val Seq(m, n) = A.shapes
    val lwork = work.shapes(0)
    val Tsize = T.shapes(0)

    val ldA = A.eltStride(1).max(1)
    val info = cb.newLocal[Int]("dgeqrf_info")
    cb.assign(info, Code.invokeScalaObject8[Int, Int, Long, Int, Long, Int, Long, Int, Int](LAPACK.getClass, "dgeqr",
      m.toI, n.toI,
      A.firstDataAddress, ldA,
      T.firstDataAddress, Tsize.toI,
      work.firstDataAddress, lwork.toI))
    val optTsize = cb.memoize(T.loadElement(FastIndexedSeq(0), cb).get.asFloat64.code.toI)
    val optLwork = cb.memoize(work.loadElement(FastIndexedSeq(0), cb).get.asFloat64.code.toI)
    cb.ifx(optTsize > Tsize.toI, cb._fatal(s"dgeqr: T too small"))
    cb.ifx(optLwork > lwork.toI, cb._fatal(s"dgeqr: work too small"))
    cb.ifx(info.cne(0), cb._fatal(s"LAPACK error dgeqr. Error code = ", info.toS))
  }

  def gemqr(cb: EmitCodeBuilder, side: String, trans: String, A: SNDArrayValue, T: SNDArrayValue, C: SNDArrayValue, work: SNDArrayValue): Unit = {
    assertMatrix(A)
    assertColMajor(cb, A)
    if (C.st.nDims == 2) assertColMajor(cb, C) else assertVector(C)
    assertVector(work, T)

    assert(side == "L" || side == "R")
    assert(trans == "T" || trans == "N")
    val Seq(l, k) = A.shapes
    val Seq(m, n) = if (C.st.nDims == 2) C.shapes else FastIndexedSeq(C.shapes(0), SizeValueStatic(1))
    if (side == "L") {
      cb.ifx(l.cne(m), cb._fatal("gemqr: invalid dimensions"))
    } else {
      cb.ifx(l.cne(n), cb._fatal("gemqr: invalid dimensions"))
    }
    val Tsize = T.shapes(0)
    val Lwork = work.shapes(0)

    val error = cb.mb.newLocal[Int]()
    val ldA = A.eltStride(1).max(1)
    val ldC = if (C.st.nDims == 2) C.eltStride(1).max(1) else m.toI
    cb.assign(error, Code.invokeScalaObject13[String, String, Int, Int, Int, Long, Int, Long, Int, Long, Int, Long, Int, Int](LAPACK.getClass, "dgemqr",
      side, trans, m.toI, n.toI, k.toI,
      A.firstDataAddress, ldA,
      T.firstDataAddress, Tsize.toI,
      C.firstDataAddress, ldC,
      work.firstDataAddress, Lwork.toI))
    cb.ifx(error.cne(0), cb._fatal("LAPACK error dgemqr. Error code = ", error.toS))
  }

  // Computes the QR factorization of A. Stores resulting factors in Q and R, overwriting A.
  def geqr_full(cb: EmitCodeBuilder, A: SNDArrayValue, Q: SNDArrayValue, R: SNDArrayValue, T: SNDArrayValue, work: SNDArrayValue): Unit = {
    val Seq(m, n) = A.shapes
    SNDArray.geqr(cb, A, T, work)
    // copy upper triangle of A0 to R
    SNDArray.copyMatrix(cb, "U", A.slice(cb, (null, n), ::), R)

    // Set Q to I
    Q.setToZero(cb)
    val i = cb.mb.newLocal[Long]("i")
    cb.forLoop(cb.assign(i, 0L), i < n, cb.assign(i, i+1), {
      Q.setElement(FastIndexedSeq(i, i), primitive(1.0), cb)
    })
    SNDArray.gemqr(cb, "L", "N", A, T, Q, work)
  }

  def tpqrt(A: SNDArrayValue, B: SNDArrayValue, T: SNDArrayValue, work: SNDArrayValue, blocksize: Value[Long], cb: EmitCodeBuilder): Unit = {
    assertMatrix(A, B)
    assertColMajor(cb, A, B)
    assertVector(work, T)

    val Seq(m, n) = B.shapes
    val nb = blocksize
    cb.ifx(nb > n || nb < 1, cb._fatal("tpqrt: invalid block size"))
    cb.ifx(T.shapes(0) < nb*n, cb._fatal("tpqrt: T too small"))
    A.assertHasShape(cb, FastIndexedSeq(n, n), "tpqrt: invalid shapes")
    cb.ifx(work.shapes(0) < nb * n, cb._fatal("tpqrt: work array too small"))

    val error = cb.mb.newLocal[Int]()
    val ldA = A.eltStride(1).max(1)
    val ldB = B.eltStride(1).max(1)
    cb.assign(error, Code.invokeScalaObject11[Int, Int, Int, Int, Long, Int, Long, Int, Long, Int, Long, Int](LAPACK.getClass, "dtpqrt",
      m.toI, n.toI, 0, nb.toI,
      A.firstDataAddress, ldA,
      B.firstDataAddress, ldB,
      T.firstDataAddress, nb.toI.max(1),
      work.firstDataAddress))
    cb.ifx(error.cne(0), cb._fatal("LAPACK error dtpqrt. Error code = ", error.toS))
  }

  def tpmqrt(side: String, trans: String, V: SNDArrayValue, T: SNDArrayValue, A: SNDArrayValue, B: SNDArrayValue, work: SNDArrayValue, blocksize: Value[Long], cb: EmitCodeBuilder): Unit = {
    assertMatrix(A, B, V)
    assertColMajor(cb, A, B, V)
    assertVector(work, T)

    assert(side == "L" || side == "R")
    assert(trans == "T" || trans == "N")
    val Seq(l, k) = V.shapes
    val Seq(m, n) = B.shapes
    val nb = blocksize
    cb.ifx(nb > k || nb < 1, cb._fatal("tpmqrt: invalid block size"))
    cb.ifx(T.shapes(0) < nb*k, cb._fatal("tpmqrt: T too small"))
    if (side == "L") {
      cb.ifx(l.cne(m), cb._fatal("tpmqrt: invalid dimensions"))
      cb.ifx(work.shapes(0) < nb * n, cb._fatal("tpmqrt: work array too small"))
      A.assertHasShape(cb, FastIndexedSeq(k, n), "tpmqrt: invalid shapes")
    } else {
      cb.ifx(l.cne(n), cb._fatal("invalid dimensions"))
      cb.ifx(work.shapes(0) < nb * m, cb._fatal("work array too small"))
      A.assertHasShape(cb, FastIndexedSeq(m, k), "tpmqrt: invalid shapes")
    }

    val error = cb.mb.newLocal[Int]()
    val ldV = V.eltStride(1).max(1)
    val ldA = A.eltStride(1).max(1)
    val ldB = B.eltStride(1).max(1)
    cb.assign(error, Code.invokeScalaObject16[String, String, Int, Int, Int, Int, Int, Long, Int, Long, Int, Long, Int, Long, Int, Long, Int](LAPACK.getClass, "dtpmqrt",
      side, trans, m.toI, n.toI, k.toI, 0, nb.toI,
      V.firstDataAddress, ldV,
      T.firstDataAddress, nb.toI.max(1),
      A.firstDataAddress, ldA,
      B.firstDataAddress, ldB,
      work.firstDataAddress))
    cb.ifx(error.cne(0), cb._fatal("LAPACK error dtpqrt. Error code = ", error.toS))
  }

  def geqrf_query(cb: EmitCodeBuilder, m: Value[Int], n: Value[Int], region: Value[Region]): Value[Int] = {
    val LWorkAddress = cb.newLocal[Long]("dgeqrf_lwork_address")
    val LWork = cb.newLocal[Int]("dgeqrf_lwork")
    val info = cb.newLocal[Int]("dgeqrf_info")
    cb.assign(LWorkAddress, region.allocate(8L, 8L))
    cb.assign(info, Code.invokeScalaObject7[Int, Int, Long, Int, Long, Long, Int, Int](LAPACK.getClass, "dgeqrf",
      m.toI, n.toI,
      0, m.toI,
      0,
      LWorkAddress, -1))
    cb.ifx(info.cne(0), cb._fatal(s"LAPACK error DGEQRF. Failed size query. Error code = ", info.toS))
    cb.assign(LWork, Region.loadDouble(LWorkAddress).toI)
    cb.memoize((LWork > 0).mux(LWork, 1))
  }

  def geqrf(cb: EmitCodeBuilder, A: SNDArrayValue, T: SNDArrayValue, work: SNDArrayValue): Unit = {
    assertMatrix(A)
    assertColMajor(cb, A)
    assertVector(T, work)

    val Seq(m, n) = A.shapes
    cb.ifx(T.shapes(0).cne(m.min(n)), cb._fatal("geqrf: T has wrong size"))
    val lwork = work.shapes(0)
    cb.ifx(lwork < n.max(1L), cb._fatal("geqrf: work has wrong size"))

    val ldA = A.eltStride(1).max(1)
    val info = cb.newLocal[Int]("dgeqrf_info")
    cb.assign(info, Code.invokeScalaObject7[Int, Int, Long, Int, Long, Long, Int, Int](LAPACK.getClass, "dgeqrf",
      m.toI, n.toI,
      A.firstDataAddress, ldA,
      T.firstDataAddress,
      work.firstDataAddress, lwork.toI))
    cb.ifx(info.cne(0), cb._fatal(s"LAPACK error DGEQRF. Error code = ", info.toS))
  }

  def orgqr(cb: EmitCodeBuilder, k: Value[Int], A: SNDArrayValue, T: SNDArrayValue, work: SNDArrayValue): Unit = {
    assertMatrix(A)
    assertColMajor(cb, A)
    assertVector(T, work)

    val Seq(m, n) = A.shapes
    cb.ifx(k < 0 || k > n.toI, cb._fatal("orgqr: invalid k"))
    cb.ifx(T.shapes(0).cne(m.min(n)), cb._fatal("orgqr: T has wrong size"))
    val lwork = work.shapes(0)
    cb.ifx(lwork < n.max(1L), cb._fatal("orgqr: work has wrong size"))

    val ldA = A.eltStride(1).max(1)
    val info = cb.newLocal[Int]("dgeqrf_info")
    cb.assign(info, Code.invokeScalaObject8[Int, Int, Int, Long, Int, Long, Long, Int, Int](LAPACK.getClass, "dorgqr",
      m.toI, n.toI, k.toI,
      A.firstDataAddress, ldA,
      T.firstDataAddress,
      work.firstDataAddress, lwork.toI))
    cb.ifx(info.cne(0), cb._fatal(s"LAPACK error DGEQRF. Error code = ", info.toS))
  }
}


trait SNDArray extends SType {
  def pType: PNDArray

  def nDims: Int

  def elementType: SType
  def elementPType: PType
  def elementEmitType: EmitType = EmitType(elementType, pType.elementType.required)

  def elementByteSize: Long

  override def _typeWithRequiredness: TypeWithRequiredness = RNDArray(elementType.typeWithRequiredness.setRequired(true).r)
}

sealed abstract class NDArrayIndex
case class ScalarIndex(i: Value[Long]) extends NDArrayIndex
case class SliceIndex(begin: Option[Value[Long]], end: Option[Value[Long]]) extends NDArrayIndex
case class SliceSize(begin: Option[Value[Long]], size: SizeValue) extends NDArrayIndex
case object ColonIndex extends NDArrayIndex

// Used to preserve static information about dimension sizes.
// If `l == r`, then we know statically that the sizes are equal, even if
// the size itself is dynamic (e.g. they share the same storage location)
// `l.ceq(r)` compares the sizes dynamically, but takes advantage of static
// knowledge to elide the comparison when possible.
sealed abstract class SizeValue extends Value[Long] {
  def ceq(other: SizeValue): Code[Boolean] = (this, other) match {
    case (SizeValueStatic(l), SizeValueStatic(r)) => const(l == r)
    case (l, r) => if (l == r) const(true) else l.get.ceq(r.get)
  }
  def cne(other: SizeValue): Code[Boolean] = (this, other) match {
    case (SizeValueStatic(l), SizeValueStatic(r)) => const(l != r)
    case (l, r) => if (l == r) const(false) else l.get.cne(r.get)
  }
}
object SizeValueDyn {
  def apply(v: Value[Long]): SizeValueDyn = new SizeValueDyn(v)
  def unapply(size: SizeValueDyn): Some[Value[Long]] = Some(size.v)
}
object SizeValueStatic {
  def apply(v: Long): SizeValueStatic = {
    assert(v >= 0)
    new SizeValueStatic(v)
  }
  def unapply(size: SizeValueStatic): Some[Long] = Some(size.v)
}
final class SizeValueDyn(val v: Value[Long]) extends SizeValue {
  def get: Code[Long] = v.get
  override def equals(other: Any): Boolean = other match {
    case SizeValueDyn(v2) => v eq v2
    case _ => false
  }
}
final class SizeValueStatic(val v: Long) extends SizeValue {
  def get: Code[Long] = const(v)
  override def equals(other: Any): Boolean = other match {
    case SizeValueStatic(v2) => v == v2
    case _ => false
  }
}

trait SNDArrayValue extends SValue {
  def st: SNDArray

  override def get: SNDArrayCode

  def loadElement(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): SValue

  def loadElementAddress(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): Code[Long]

  def shapes: IndexedSeq[SizeValue]

  def shapeStruct(cb: EmitCodeBuilder): SBaseStructValue

  def strides: IndexedSeq[Value[Long]]

  def eltStride(i: Int): Code[Int] = st.elementByteSize match {
    case 4 => strides(i).toI >> 2
    case 8 => strides(i).toI >> 3
    case eltSize => strides(i).toI / eltSize.toInt
  }

  def firstDataAddress: Value[Long]

  def outOfBounds(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): Code[Boolean] = {
    val shape = this.shapes
    val outOfBounds = cb.newLocal[Boolean]("sndarray_out_of_bounds", false)

    (0 until st.nDims).foreach { dimIndex =>
      cb.assign(outOfBounds, outOfBounds || (indices(dimIndex) >= shape(dimIndex)))
    }
    outOfBounds
  }

  def assertInBounds(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder, errorId: Int): Unit = {
    val shape = this.shapes
    for (dimIndex <- 0 until st.nDims) {
      cb.ifx(indices(dimIndex) >= shape(dimIndex), {
        cb._fatalWithError(errorId,
          "Index ", indices(dimIndex).toS,
          s" is out of bounds for axis $dimIndex with size ",
          shape(dimIndex).toS)
      })
    }
  }

  def sameShape(cb: EmitCodeBuilder, other: SNDArrayValue): Code[Boolean] =
    hasShape(cb, other.shapes)

  def hasShape(cb: EmitCodeBuilder, otherShape: IndexedSeq[SizeValue]): Code[Boolean] = {
    var b: Code[Boolean] = const(true)
    val shape = this.shapes
    assert(shape.length == otherShape.length)

    (shape, otherShape).zipped.foreach { (s1, s2) =>
      b = s1.ceq(s2)
    }
    b
  }

  def assertHasShape(cb: EmitCodeBuilder, otherShape: IndexedSeq[SizeValue], msg: Code[String]*) =
    if (!hasShapeStatic(otherShape))
      cb.ifx(!hasShape(cb, otherShape), cb._fatal(msg: _*))

  // True IFF shape can be proven equal to otherShape statically
  def hasShapeStatic(otherShape: IndexedSeq[SizeValue]): Boolean =
    shapes == otherShape

  def hasShapeStatic(otherShape: SizeValue*): Boolean =
    hasShapeStatic(otherShape.toFastIndexedSeq)

  def isVector: Boolean = shapes.length == 1

  // ensure coerceToShape(cb, otherShape).hasShapeStatic(otherShape)
  // Inserts any necessary dynamic assertions
  def coerceToShape(cb: EmitCodeBuilder, otherShape: IndexedSeq[SizeValue]): SNDArrayValue

  def coerceToShape(cb: EmitCodeBuilder, otherShape: SizeValue*): SNDArrayValue =
    coerceToShape(cb, otherShape.toFastIndexedSeq)

  def contiguousDimensions(cb: EmitCodeBuilder): Value[Int] = {
    val tmp = cb.mb.newLocal[Int]("NDArray_setToZero_tmp")
    val contiguousDims = cb.mb.newLocal[Int]("NDArray_setToZero_contigDims")

    cb.assign(tmp, 1)

    // Find largest prefix of dimensions which are stored contiguously.
    def contigDimsRecur(i: Int): Unit =
      if (i < st.nDims) {
        cb.ifx(tmp.ceq(eltStride(i)), {
          cb.assign(tmp, tmp * shapes(i).toI)
          contigDimsRecur(i+1)
        }, {
          cb.assign(contiguousDims, i)
        })
      } else {
        cb.assign(contiguousDims, st.nDims)
      }

    contigDimsRecur(0)

    contiguousDims
  }

  def setToZero(cb: EmitCodeBuilder): Unit

  def setElement(indices: IndexedSeq[Value[Long]], value: SValue, cb: EmitCodeBuilder): Unit = {
    val eltType = st.pType.elementType.asInstanceOf[PPrimitive]
    eltType.storePrimitiveAtAddress(cb, loadElementAddress(indices, cb), value)
  }

  def coiterateMutate(cb: EmitCodeBuilder, region: Value[Region], arrays: (SNDArrayValue, String)*)(body: IndexedSeq[SValue] => SValue): Unit =
    coiterateMutate(cb, region, false, arrays: _*)(body)

  def coiterateMutate(cb: EmitCodeBuilder, region: Value[Region], deepCopy: Boolean, arrays: (SNDArrayValue, String)*)(body: IndexedSeq[SValue] => SValue): Unit = {
    val indexVars = Array.tabulate(st.nDims)(i => s"i$i").toFastIndexedSeq
    val indices = Array.range(0, st.nDims).toFastIndexedSeq
    coiterateMutate(cb, region, deepCopy, indexVars, indices, arrays.map { case (array, name) => (array, indices, name) }: _*)(body)
  }

  def coiterateMutate(cb: EmitCodeBuilder, region: Value[Region], indexVars: IndexedSeq[String], destIndices: IndexedSeq[Int], arrays: (SNDArrayValue, IndexedSeq[Int], String)*)(body: IndexedSeq[SValue] => SValue): Unit =
    coiterateMutate(cb, region, false, indexVars, destIndices, arrays: _*)(body)

  // Note: to iterate through an array in column major order, make sure the indices are in ascending order. E.g.
  // A.coiterateMutate(cb, region, IndexedSeq("i", "j"), IndexedSeq((A, IndexedSeq(0, 1), "A"), (B, IndexedSeq(0, 1), "B")), {
  //   SCode.add(cb, a, b)
  // })
  // computes A += B.
  def coiterateMutate(
    cb: EmitCodeBuilder,
    region: Value[Region],
    deepCopy: Boolean,
    indexVars: IndexedSeq[String],
    destIndices: IndexedSeq[Int],
    arrays: (SNDArrayValue, IndexedSeq[Int], String)*
  )(body: IndexedSeq[SValue] => SValue
  ): Unit

  def _slice(cb: EmitCodeBuilder, indices: IndexedSeq[NDArrayIndex]): SNDArraySliceValue = {
    val shapeX = shapes
    val stridesX = strides
    val shapeBuilder = mutable.ArrayBuilder.make[SizeValue]
    val stridesBuilder = mutable.ArrayBuilder.make[Value[Long]]

    for (i <- indices.indices) indices(i) match {
      case ScalarIndex(j) =>
        cb.ifx(j < 0 || j >= shapeX(i), cb._fatal("Index out of bounds"))
      case SliceIndex(Some(begin), Some(end)) =>
        cb.ifx(begin < 0 || end > shapeX(i) || begin > end, cb._fatal("Index out of bounds"))
        val s = cb.newLocal[Long]("slice_size", end - begin)
        shapeBuilder += SizeValueDyn(s)
        stridesBuilder += stridesX(i)
      case SliceIndex(None, Some(end: SizeValue)) =>
        cb.ifx(end > shapeX(i) || end < 0, cb._fatal("Index out of bounds"))
        shapeBuilder += end
        stridesBuilder += stridesX(i)
      case SliceIndex(None, Some(end)) =>
        cb.ifx(end > shapeX(i) || end < 0, cb._fatal("Index out of bounds"))
        shapeBuilder += SizeValueDyn(end)
        stridesBuilder += stridesX(i)
      case SliceIndex(Some(begin), None) =>
        val end = shapeX(i)
        cb.ifx(begin < 0 || begin > end, cb._fatal("Index out of bounds"))
        val s = cb.newLocal[Long]("slice_size", end - begin)
        shapeBuilder += SizeValueDyn(s)
        stridesBuilder += stridesX(i)
      case SliceIndex(None, None) =>
        shapeBuilder += shapeX(i)
        stridesBuilder += stridesX(i)
      case SliceSize(None, size) =>
        cb.ifx(size >= shapeX(i), cb._fatal("Index out of bounds") )
        shapeBuilder += size
        stridesBuilder += stridesX(i)
      case SliceSize(Some(begin), size) =>
        cb.ifx(begin < 0 || begin + size > shapeX(i), cb._fatal("Index out of bounds") )
        shapeBuilder += size
        stridesBuilder += stridesX(i)
      case ColonIndex =>
        shapeBuilder += shapeX(i)
        stridesBuilder += stridesX(i)
    }
    val newShape = shapeBuilder.result()
    val newStrides = stridesBuilder.result()

    val firstElementIndices = indices.map {
      case ScalarIndex(j) => j
      case SliceIndex(Some(begin), _) => begin
      case SliceIndex(None, _) => const(0L)
      case ColonIndex => const(0L)
    }

    val newFirstDataAddress = cb.newLocal[Long]("slice_ptr", loadElementAddress(firstElementIndices, cb))

    val newSType = SNDArraySlice(PCanonicalNDArray(st.pType.elementType, newShape.size, st.pType.required))

    new SNDArraySliceValue(newSType, newShape, newStrides, newFirstDataAddress)
  }

  def slice(cb: EmitCodeBuilder, indices: Any*): SNDArraySliceValue = {
    val parsedIndices: IndexedSeq[NDArrayIndex] = indices.map {
      case _: ::.type => ColonIndex
      case i: Value[_] => ScalarIndex(i.asInstanceOf[Value[Long]])
      case i: Code[_] => ScalarIndex(cb.memoize(coerce[Long](i)))
      case (_begin, _end) =>
        val parsedBegin = _begin match {
          case begin: Value[_] => Some(begin.asInstanceOf[Value[Long]])
          case begin: Code[_] => Some(cb.memoize(coerce[Long](begin)))
          case begin: Int => Some(const(begin.toLong))
          case begin: Long => Some(const(begin))
          case null => None
        }
        val parsedEnd = _end match {
          case end: Value[_] => Some(end.asInstanceOf[Value[Long]])
          case end: Code[_] => Some(cb.memoize(coerce[Long](end)))
          case end: Int => Some(const(end.toLong))
          case end: Long => Some(const(end))
          case null => None
        }
        SliceIndex(parsedBegin, parsedEnd)
    }.toIndexedSeq
    _slice(cb, parsedIndices)
  }
}

trait SNDArraySettable extends SNDArrayValue with SSettable

trait SNDArrayCode extends SCode {
  def st: SNDArray

  def memoize(cb: EmitCodeBuilder, name: String): SNDArrayValue
}

class LocalWhitening(cb: EmitCodeBuilder, vecSize: SizeValue, _w: Value[Long], chunksize: Value[Long], _blocksize: Value[Long], region: Value[Region], normalizeAfterWhitening: Boolean) {
  val m = vecSize
  val w = SizeValueDyn(_w)
  val b = SizeValueDyn(chunksize)
  val wpb = SizeValueDyn(cb.memoize(w+b))

  val curSize = cb.newField[Long]("curSize", 0)
  val pivot = cb.newField[Long]("pivot", 0)

  val matType = PCanonicalNDArray(PFloat64Required, 2)
  val vecType = PCanonicalNDArray(PFloat64Required, 1)

  val (tsize, worksize) = SNDArray.geqr_query(cb, m, chunksize, region)

  val Q = matType.constructUnintialized(FastIndexedSeq(m, w), cb, region)
  val R = matType.constructUnintialized(FastIndexedSeq(w, w), cb, region)
  val work1 = matType.constructUnintialized(FastIndexedSeq(wpb, wpb), cb, region)
  val work2 = matType.constructUnintialized(FastIndexedSeq(wpb, b), cb, region)
  val Rtemp = matType.constructUnintialized(FastIndexedSeq(wpb, wpb), cb, region)
  val Qtemp = matType.constructUnintialized(FastIndexedSeq(m, b), cb, region)
  val Qtemp2 = matType.constructUnintialized(FastIndexedSeq(m, w), cb, region)
  val blocksize = cb.memoize(_blocksize.min(w))
  val work3len = SizeValueDyn(cb.memoize(worksize.max(blocksize*m)))
  val work3: SNDArrayValue = vecType.constructUnintialized(FastIndexedSeq(work3len), cb, region)
  val Tlen = SizeValueDyn(cb.memoize(tsize.max(blocksize*wpb)))
  val T: SNDArrayValue = vecType.constructUnintialized(FastIndexedSeq(Tlen), cb, region)

  def reset(cb: EmitCodeBuilder): Unit = {
    cb.assign(curSize, 0L)
    cb.assign(pivot, 0L)
  }

  // Pre: A1 is current window, A2 is next window, [Q1 Q2] R = [A1 A2] is qr fact
  // Post: W contains locally whitened A2, Qout R[-w:, -w:] = A2[:, -w:] is qr fact
  def whitenBase(cb: EmitCodeBuilder,
    Q1: SNDArrayValue, Q2: SNDArrayValue, Qout: SNDArrayValue,
    R: SNDArrayValue, W: SNDArrayValue,
    work1: SNDArrayValue, work2: SNDArrayValue,
    blocksize: Value[Long]
  ): Unit = {
    SNDArray.assertMatrix(Q1, Q2, Qout, R, work1, work2)
    SNDArray.assertColMajor(cb, Q1, Q2, Qout, R, work1, work2)

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
      val w1col = work1.slice(cb, ::, wpi)
      val w2col = work2.slice(cb, ::, i)
      SNDArray.copyVector(cb, w1col, w2col)
      if (!normalizeAfterWhitening) {
        SNDArray.scale(cb, R.loadElement(FastIndexedSeq(wpi, wpi), cb), w2col)
      }

      // work3 > blocksize * (w+n - i+1) < blocksize * (w+n)
      SNDArray.tpqrt(R.slice(cb, (i+1, null), (i+1, null)), R.slice(cb, (i, i+1), (i+1, null)), T, work3, blocksize, cb)
      SNDArray.tpmqrt("R", "N", R.slice(cb, (i, i+1), (i+1, null)), T, work1.slice(cb, ::, (i+1, null)), work1.slice(cb, ::, (i, i+1)), work3, blocksize, cb)
    })

    // W = [Q1 Q2] work2 is locally whitened A2
    SNDArray.gemm(cb, "N", "N", 1.0, Q1, work2.slice(cb, (null, w), ::), 0.0, W)
    SNDArray.gemm(cb, "N", "N", 1.0, Q2, work2.slice(cb, (w, null), ::), 1.0, W)

    // Qout = [Q1 Q2] work1, Qout R[n:w+n, n:w+n] = A2[:, n-w:n] is qr fact
    SNDArray.gemm(cb, "N", "N", 1.0, Q1, work1.slice(cb, (null, w), (n, null)), 0.0, Qout)
    SNDArray.gemm(cb, "N", "N", 1.0, Q2, work1.slice(cb, (w, null), (n, null)), 1.0, Qout)
  }

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
    cb.ifx(m <= w, cb._fatal("qr_pivot: m <= w"))
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
      SNDArray.tpmqrt("R", "N", R.slice(cb, r1, r2), T, Q.slice(cb, ::, r2), Q.slice(cb, ::, r1), work3, b2, cb)
    })
    cb.ifx(p0 > 0, {
      SNDArray.tpqrt(R.slice(cb, r0, r0), R.slice(cb, r1, r0), T, work3, b0, cb)
      SNDArray.tpmqrt("L", "T", R.slice(cb, r1, r0), T, R.slice(cb, r0, r1), R.slice(cb, r1, r1), work3, b0, cb)
      SNDArray.tpmqrt("R", "N", R.slice(cb, r1, r0), T, Q.slice(cb, ::, r0), Q.slice(cb, ::, r1), work3, b0, cb)
    })
    SNDArray.geqrt(R.slice(cb, r1, r1), T, work3, b1, cb)
    SNDArray.gemqrt("R", "N", R.slice(cb, r1, r1), T, Q.slice(cb, ::, r1), work3, b1, cb)
  }

  // Pre: Q R = A0 is qr fact of current window, A contains next window
  // Post: A contains A_orig whitened, Q R = A_orig
  def whitenNonrecur(cb: EmitCodeBuilder,
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
    whitenBase(cb, Q, Qtemp, Qtemp2, Rtemp, A, work1, work2, blocksize)

    // copy upper triangle of Rtemp[n:w+n, n:w+n] to R
    SNDArray.copyMatrix(cb, "U", Rtemp.slice(cb, (n, w+n), (n, w+n)), R)
    // copy Qtemp2 to Q
    SNDArray.copyMatrix(cb, " ", Qtemp2, Q)
    // now Q R = A_orig[::, n-w:n]
  }

  // Pre: Let b = A.shapes(A, 1), Q1 = Q[:, 0:p], Q2 = Q[:, p:p+b], Q3 = Q[:, p+b:w]
  // * [Q2 Q3 Q1] [R22 R23 R21; 0 R33 R31; 0 0 R11] = [A2 A3 A1] is a qr fact
  // Post:
  // * [Q3 Q1 Q2] [R33 R31 R32; 0 R11 R12; 0 0 R22] = [A3 A1 A_orig] is a qr fact
  // * A contains whitened A_orig
  def whitenStep(cb: EmitCodeBuilder,
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
    val Q1 = Q.slice(cb, ::, r1)
    val Q2 = Q.slice(cb, ::, r2)
    val Q3 = Q.slice(cb, ::, r3)

    // Orthogonalize against Q3
    SNDArray.gemm(cb, "T", "N", Q3, A, R32)
    SNDArray.gemm(cb, "N", "N", -1.0, Q3, R32, 1.0, A)
    // Orthogonalize against Q1
    SNDArray.gemm(cb, "T", "N", Q1, A, R12)
    SNDArray.gemm(cb, "N", "N", -1.0, Q1, R12, 1.0, A)

    // Now A = A_orig - Q3 R32 - Q1 R12
    whitenNonrecur(cb, Q2, R22, A, Qtemp, Qtemp2, Rtemp, work1, work2, blocksize)
    // now A contains A_orig - Q3 R32 - Q1 R12 whitened against A2
    // and Q2 R22 = A = A_orig - Q3 R32 - Q1 R12
    // so A_orig = Q3 R32 + Q1 R12 + Q2 R22
  }

  def whitenBlock(cb: EmitCodeBuilder, _A: SNDArrayValue): Unit = {
    val b = _A.shapes(1)

    val A = _A.coerceToShape(cb, m, b)
    cb.ifx(b > chunksize, cb._fatal("whitenBlock: A too large, found ", b.toS, ", expected ", chunksize.toS))

    cb.ifx(curSize < w, {
      // Orthogonalize against existing Q
      val Rslice = R.slice(cb, (null, curSize), (curSize, curSize + b))
      val Qslice = Q.slice(cb, ::, (null, curSize))
      // Rslice = Q' A
      SNDArray.gemm(cb, "T", "N", Qslice, A, Rslice)
      // A = A - Q Rslice
      SNDArray.gemm(cb, "N", "N", -1.0, Qslice, Rslice, 1.0, A)

      // Compute QR fact of A; store R fact in Rtemp[r1, r1], Q fact in Qtemp
      val Rslice2 = R.slice(cb, (curSize, curSize + b), (curSize, curSize + b))
      val Qslice2 = Q.slice(cb, ::, (curSize, curSize + b))
      SNDArray.geqr_full(cb, A, Qslice2, Rslice2, T, work3)

      // Copy whitened A back to A
      val j = cb.newLocal[Long]("j")
      cb.forLoop(cb.assign(j, 0L), j < b, cb.assign(j, j+1), {
        val Acol = A.slice(cb, ::, j)
        SNDArray.copyVector(cb, Qslice2.slice(cb, ::, j), Acol)
        SNDArray.scale(cb, Rslice2.loadElement(FastIndexedSeq(j, j), cb), Acol)
      })

      cb.assign(curSize, curSize + b)
    }, {
      cb.ifx(curSize.cne(w), cb._fatal("whitenBlock: initial blocks didn't evenly divide window size"))

      val bb = SizeValueDyn(cb.memoize(b*2))
      whitenStep(cb,
        Q, R, pivot, A,
        Qtemp.slice(cb, ::, (null, b)),
        Qtemp2.slice(cb, ::, (null, b)),
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
    val Qslice = Q.slice(cb, ::, (null, b))
    SNDArray.geqr_full(cb, A, Qslice, Rslice, T, work3)
    cb.assign(curSize, b)
  }
}

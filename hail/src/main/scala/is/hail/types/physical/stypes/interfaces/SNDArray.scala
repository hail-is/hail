package is.hail.types.physical.stypes.interfaces

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.linalg.{BLAS, LAPACK}
import is.hail.types.{RNDArray, TypeWithRequiredness}
import is.hail.types.physical._
import is.hail.types.physical.stypes.{EmitType, SSettable, SType, SValue}
import is.hail.types.physical.stypes.concrete.{SNDArraySlice, SNDArraySliceValue}
import is.hail.types.physical.stypes.primitives.SInt64Value
import is.hail.types.virtual.TInt32
import is.hail.utils.{toRichIterable, valueToRichCodeRegion, FastSeq}

import scala.collection.mutable

object SNDArray {
  def numElements(shape: IndexedSeq[Value[Long]]): Code[Long] =
    shape.foldLeft(1L: Code[Long])(_ * _)

  // Column major order
  def forEachIndexColMajor(
    cb: EmitCodeBuilder,
    shape: IndexedSeq[Value[Long]],
    context: String,
  )(
    f: (EmitCodeBuilder, IndexedSeq[Value[Long]]) => Unit
  ): Unit =
    forEachIndexWithInitAndIncColMajor(
      cb,
      shape,
      shape.map(_ => (cb: EmitCodeBuilder) => ()),
      shape.map(_ => (cb: EmitCodeBuilder) => ()),
      context,
    )(f)

  def coiterate(
    cb: EmitCodeBuilder,
    arrays: (SNDArrayValue, String)*
  )(
    body: IndexedSeq[SValue] => Unit
  ): Unit = {
    if (arrays.isEmpty) return
    val indexVars = Array.tabulate(arrays(0)._1.st.nDims)(i => s"i$i").toFastSeq
    val indices = Array.range(0, arrays(0)._1.st.nDims).toFastSeq
    coiterate(cb, indexVars, arrays.map { case (array, name) => (array, indices, name) }: _*)(body)
  }

  /* Note: to iterate through an array in column major order, make sure the indices are in ascending
   * order. E.g. */
  // A.coiterate(cb, IndexedSeq("i", "j"), (A, IndexedSeq(0, 1), "A"), (B, IndexedSeq(0, 1), "B"), {
  //   SCode.add(cb, a, b)
  // })
  // computes A += B.
  def coiterate(
    cb: EmitCodeBuilder,
    indexVars: IndexedSeq[String],
    arrays: (SNDArrayValue, IndexedSeq[Int], String)*
  )(
    body: IndexedSeq[SValue] => Unit
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
  )(
    body: IndexedSeq[Value[Long]] => Unit
  ): Unit = {
    val indexSizes = new Array[Settable[Int]](indexVars.length)
    val indexCoords = Array.tabulate(indexVars.length)(i => cb.newLocal[Int](indexVars(i)))

    case class ArrayInfo(
      array: SNDArrayValue,
      strides: IndexedSeq[Value[Long]],
      pos: IndexedSeq[Settable[Long]],
      indexToDim: Map[Int, Int],
      name: String,
    )

    val info = arrays.toIndexedSeq.map { case (array, indices, name) =>
      for (idx <- indices) assert(idx < indexVars.length && idx >= 0)
      // FIXME: relax this assumption to handle transposing, non-column major
      for (i <- 0 until indices.length - 1) assert(indices(i) < indices(i + 1))
      assert(indices.length == array.st.nDims)

      val shape = array.shapes
      for (i <- indices.indices) {
        val idx = indices(i)
        if (indexSizes(idx) == null) {
          indexSizes(idx) = cb.newLocal[Int](s"${indexVars(idx)}_max")
          cb.assign(indexSizes(idx), shape(i).toI)
        } else {
          cb.if_(
            indexSizes(idx).cne(shape(i).toI),
            cb._fatal(s"${indexVars(idx)} indexes incompatible dimensions"),
          )
        }
      }
      val strides = array.strides
      val pos = Array.tabulate(array.st.nDims + 1)(i => cb.newLocal[Long](s"$name$i"))
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
          for (n <- arrays.indices)
            if (info(n).indexToDim.contains(idx)) {
              val i = info(n).indexToDim(idx)
              // FIXME: assumes array's indices in ascending order
              cb.assign(info(n).pos(i), info(n).pos(i + 1))
            }
        }
        def increment(): Unit = {
          cb.assign(coord, coord + 1)
          for (n <- arrays.indices)
            if (info(n).indexToDim.contains(idx)) {
              val i = info(n).indexToDim(idx)
              cb.assign(info(n).pos(i), info(n).pos(i) + info(n).strides(i))
            }
        }

        cb.for_(init(), coord < indexSizes(idx), increment(), recurLoopBuilder(idx - 1))
      }
    }

    for (n <- arrays.indices)
      cb.assign(info(n).pos(info(n).array.st.nDims), info(n).array.firstDataAddress)
    recurLoopBuilder(indexVars.length - 1)
  }

  // Column major order
  def forEachIndexWithInitAndIncColMajor(
    cb: EmitCodeBuilder,
    shape: IndexedSeq[Value[Long]],
    inits: IndexedSeq[EmitCodeBuilder => Unit],
    incrementers: IndexedSeq[EmitCodeBuilder => Unit],
    context: String,
  )(
    f: (EmitCodeBuilder, IndexedSeq[Value[Long]]) => Unit
  ): Unit = {

    val indices = Array.tabulate(shape.length) { dimIdx =>
      cb.newLocal[Long](s"${context}_foreach_dim_$dimIdx", 0L)
    }

    def recurLoopBuilder(dimIdx: Int, innerLambda: () => Unit): Unit = {
      if (dimIdx == shape.length) {
        innerLambda()
      } else {
        val dimVar = indices(dimIdx)

        recurLoopBuilder(
          dimIdx + 1,
          () => {
            cb.for_(
              {
                inits(dimIdx)(cb)
                cb.assign(dimVar, 0L)
              },
              dimVar < shape(dimIdx), {
                incrementers(dimIdx)(cb)
                cb.assign(dimVar, dimVar + 1L)
              },
              innerLambda(),
            )
          },
        )
      }
    }

    val body = () => f(cb, indices)

    recurLoopBuilder(0, body)
  }

  // Row major order
  def forEachIndexRowMajor(
    cb: EmitCodeBuilder,
    shape: IndexedSeq[Value[Long]],
    context: String,
  )(
    f: (EmitCodeBuilder, IndexedSeq[Value[Long]]) => Unit
  ): Unit =
    forEachIndexWithInitAndIncRowMajor(
      cb,
      shape,
      shape.map(_ => (cb: EmitCodeBuilder) => ()),
      shape.map(_ => (cb: EmitCodeBuilder) => ()),
      context,
    )(f)

  // Row major order
  def forEachIndexWithInitAndIncRowMajor(
    cb: EmitCodeBuilder,
    shape: IndexedSeq[Value[Long]],
    inits: IndexedSeq[EmitCodeBuilder => Unit],
    incrementers: IndexedSeq[EmitCodeBuilder => Unit],
    context: String,
  )(
    f: (EmitCodeBuilder, IndexedSeq[Value[Long]]) => Unit
  ): Unit = {

    val indices = Array.tabulate(shape.length) { dimIdx =>
      cb.newLocal[Long](s"${context}_foreach_dim_$dimIdx", 0L)
    }

    def recurLoopBuilder(dimIdx: Int, innerLambda: () => Unit): Unit = {
      if (dimIdx == -1) {
        innerLambda()
      } else {
        val dimVar = indices(dimIdx)

        recurLoopBuilder(
          dimIdx - 1,
          () => {
            cb.for_(
              {
                inits(dimIdx)(cb)
                cb.assign(dimVar, 0L)
              },
              dimVar < shape(dimIdx), {
                incrementers(dimIdx)(cb)
                cb.assign(dimVar, dimVar + 1L)
              },
              innerLambda(),
            )
          },
        )
      }
    }

    val body = () => f(cb, indices)

    recurLoopBuilder(shape.length - 1, body)
  }

  // Column major order
  def unstagedForEachIndex(shape: IndexedSeq[Long])(f: IndexedSeq[Long] => Unit): Unit = {

    val indices = Array.tabulate(shape.length)(dimIdx => 0L)

    def recurLoopBuilder(dimIdx: Int, innerLambda: () => Unit): Unit = {
      if (dimIdx == shape.length) {
        innerLambda()
      } else {

        recurLoopBuilder(
          dimIdx + 1,
          () =>
            (0 until shape(dimIdx).toInt).foreach { _ =>
              innerLambda()
              indices(dimIdx) += 1
            },
        )
      }
    }

    val body = () => f(indices)

    recurLoopBuilder(0, body)
  }

  def assertMatrix(nds: SNDArrayValue*): Unit =
    for (nd <- nds) assert(nd.st.nDims == 2)

  def assertVector(nds: SNDArrayValue*): Unit =
    for (nd <- nds) assert(nd.st.nDims == 1)

  def assertColMajor(cb: EmitCodeBuilder, caller: String, nds: SNDArrayValue*): Unit =
    for (nd <- nds)
      cb.if_(
        nd.strides(0).cne(nd.st.pType.elementType.byteSize),
        cb._fatal(
          s"$caller requires column major: found row stride ",
          nd.strides(0).toS,
          ", expected ",
          nd.st.pType.elementType.byteSize.toString,
        ),
      )

  def copyVector(cb: EmitCodeBuilder, X: SNDArrayValue, Y: SNDArrayValue): Unit = {
    val Seq(n) = X.shapes

    Y.assertHasShape(
      cb,
      FastSeq(n),
      "copy: vectors have different sizes: ",
      Y.shapes(0).toS,
      ", ",
      n.toS,
    )
    val ldX = X.eltStride(0).max(1)
    val ldY = Y.eltStride(0).max(1)
    cb += Code.invokeScalaObject5[Int, Long, Int, Long, Int, Unit](
      BLAS.getClass,
      "dcopy",
      n.toI,
      X.firstDataAddress,
      ldX,
      Y.firstDataAddress,
      ldY,
    )
  }

  def copyMatrix(cb: EmitCodeBuilder, uplo: String, X: SNDArrayValue, Y: SNDArrayValue): Unit = {
    val Seq(m, n) = X.shapes
    Y.assertHasShape(cb, FastSeq(m, n), "copyMatrix: matrices have different shapes")
    val ldX = X.eltStride(1).max(1)
    val ldY = Y.eltStride(1).max(1)
    cb += Code.invokeScalaObject7[String, Int, Int, Long, Int, Long, Int, Unit](
      LAPACK.getClass,
      "dlacpy",
      uplo,
      m.toI,
      n.toI,
      X.firstDataAddress,
      ldX,
      Y.firstDataAddress,
      ldY,
    )
  }

  def scale(cb: EmitCodeBuilder, alpha: SValue, X: SNDArrayValue): Unit =
    scale(cb, alpha.asFloat64.value, X)

  def scale(cb: EmitCodeBuilder, alpha: Value[Double], X: SNDArrayValue): Unit = {
    val Seq(n) = X.shapes
    val ldX = X.eltStride(0).max(1)
    cb += Code.invokeScalaObject4[Int, Double, Long, Int, Unit](
      BLAS.getClass,
      "dscal",
      n.toI,
      alpha,
      X.firstDataAddress,
      ldX,
    )
  }

  def gemv(cb: EmitCodeBuilder, trans: String, A: SNDArrayValue, X: SNDArrayValue, Y: SNDArrayValue)
    : Unit =
    gemv(cb, trans, 1.0, A, X, 1.0, Y)

  def gemv(
    cb: EmitCodeBuilder,
    trans: String,
    alpha: Value[Double],
    A: SNDArrayValue,
    X: SNDArrayValue,
    beta: Value[Double],
    Y: SNDArrayValue,
  ): Unit = {
    assertMatrix(A)
    val Seq(m, n) = A.shapes
    val errMsg = "gemv: incompatible dimensions"
    if (trans == "N") {
      X.assertHasShape(cb, FastSeq(n), errMsg)
      Y.assertHasShape(cb, FastSeq(m), errMsg)
    } else {
      X.assertHasShape(cb, FastSeq(m), errMsg)
      Y.assertHasShape(cb, FastSeq(n), errMsg)
    }
    assertColMajor(cb, "gemv", A)

    val ldA = A.eltStride(1).max(1)
    val ldX = X.eltStride(0).max(1)
    val ldY = Y.eltStride(0).max(1)
    cb += Code.invokeScalaObject11[
      String,
      Int,
      Int,
      Double,
      Long,
      Int,
      Long,
      Int,
      Double,
      Long,
      Int,
      Unit,
    ](
      BLAS.getClass,
      "dgemv",
      trans,
      m.toI,
      n.toI,
      alpha,
      A.firstDataAddress,
      ldA,
      X.firstDataAddress,
      ldX,
      beta,
      Y.firstDataAddress,
      ldY,
    )
  }

  def gemm(
    cb: EmitCodeBuilder,
    tA: String,
    tB: String,
    A: SNDArrayValue,
    B: SNDArrayValue,
    C: SNDArrayValue,
  ): Unit =
    gemm(cb, tA, tB, 1.0, A, B, 0.0, C)

  def gemm(
    cb: EmitCodeBuilder,
    tA: String,
    tB: String,
    alpha: Value[Double],
    A: SNDArrayValue,
    B: SNDArrayValue,
    beta: Value[Double],
    C: SNDArrayValue,
  ): Unit = {
    assertMatrix(A, B, C)
    val Seq(m, n) = C.shapes
    val k = if (tA == "N") A.shapes(1) else A.shapes(0)
    val errMsg = "gemm: incompatible matrix dimensions"

    if (tA == "N")
      A.assertHasShape(cb, FastSeq(m, k), errMsg)
    else
      A.assertHasShape(cb, FastSeq(k, m), errMsg)
    if (tB == "N")
      B.assertHasShape(cb, FastSeq(k, n), errMsg)
    else
      B.assertHasShape(cb, FastSeq(n, k), errMsg)
    assertColMajor(cb, "gemm", A, B, C)

    val ldA = A.eltStride(1).max(1)
    val ldB = B.eltStride(1).max(1)
    val ldC = C.eltStride(1).max(1)
    cb += Code.invokeScalaObject13[
      String,
      String,
      Int,
      Int,
      Int,
      Double,
      Long,
      Int,
      Long,
      Int,
      Double,
      Long,
      Int,
      Unit,
    ](
      BLAS.getClass,
      "dgemm",
      tA,
      tB,
      m.toI,
      n.toI,
      k.toI,
      alpha,
      A.firstDataAddress,
      ldA,
      B.firstDataAddress,
      ldB,
      beta,
      C.firstDataAddress,
      ldC,
    )
  }

  def trmm(
    cb: EmitCodeBuilder,
    side: String,
    uplo: String,
    transA: String,
    diag: String,
    alpha: Value[Double],
    A: SNDArrayValue,
    B: SNDArrayValue,
  ): Unit = {
    assertMatrix(A, B)
    assertColMajor(cb, "trmm", A, B)

    val Seq(m, n) = B.shapes
    val Seq(a0, a1) = A.shapes
    cb.if_(a1.cne(if (side == "left") m else n), cb._fatal("trmm: incompatible matrix dimensions"))
    // Elide check in the common case that we statically know A is square
    if (a0 != a1)
      cb.if_(a0 < a1, cb._fatal("trmm: A has fewer rows than cols: ", a0.toS, ", ", a1.toS))

    val ldA = A.eltStride(1).max(1)
    val ldB = B.eltStride(1).max(1)
    cb += Code.invokeScalaObject11[
      String,
      String,
      String,
      String,
      Int,
      Int,
      Double,
      Long,
      Int,
      Long,
      Int,
      Unit,
    ](
      BLAS.getClass,
      "dtrmm",
      side,
      uplo,
      transA,
      diag,
      m.toI,
      n.toI,
      alpha,
      A.firstDataAddress,
      ldA,
      B.firstDataAddress,
      ldB,
    )
  }

  def geqrt(
    A: SNDArrayValue,
    T: SNDArrayValue,
    work: SNDArrayValue,
    blocksize: Value[Long],
    cb: EmitCodeBuilder,
  ): Unit = {
    if (A.st.nDims == 2) assertColMajor(cb, "geqrt", A) else assertVector(A)
    assertVector(work, T)

    val Seq(m, n) = if (A.st.nDims == 2) A.shapes else FastSeq(A.shapes(0), SizeValueStatic(1))
    val nb = blocksize
    val min = cb.memoize(m.min(n))
    cb.if_((nb > min && min > 0) || nb < 1, cb._fatal("geqrt: invalid block size: ", nb.toS))
    cb.if_(T.shapes(0) < nb * (m.min(n)), cb._fatal("geqrt: T too small"))
    cb.if_(work.shapes(0) < nb * n, cb._fatal("geqrt: work array too small"))

    val error = cb.mb.newLocal[Int]()
    val ldA = if (A.st.nDims == 2) A.eltStride(1).max(1) else m.toI
    cb.assign(
      error,
      Code.invokeScalaObject8[Int, Int, Int, Long, Int, Long, Int, Long, Int](
        LAPACK.getClass,
        "dgeqrt",
        m.toI,
        n.toI,
        nb.toI,
        A.firstDataAddress,
        ldA,
        T.firstDataAddress,
        nb.toI.max(1),
        work.firstDataAddress,
      ),
    )
    cb.if_(error.cne(0), cb._fatal("LAPACK error dtpqrt. Error code = ", error.toS))
  }

  def gemqrt(
    side: String,
    trans: String,
    V: SNDArrayValue,
    T: SNDArrayValue,
    C: SNDArrayValue,
    work: SNDArrayValue,
    blocksize: Value[Long],
    cb: EmitCodeBuilder,
  ): Unit = {
    assertMatrix(V)
    assertColMajor(cb, "gemqrt", V)
    if (C.st.nDims == 2) assertColMajor(cb, "gemqrt", C) else assertVector(C)
    assertVector(work, T)

    assert(side == "L" || side == "R")
    assert(trans == "T" || trans == "N")
    val Seq(l, k) = V.shapes
    val Seq(m, n) = if (C.st.nDims == 2) C.shapes else FastSeq(C.shapes(0), SizeValueStatic(1))
    val nb = blocksize
    cb.if_((nb > k && k > 0) || nb < 1, cb._fatal("gemqrt: invalid block size: ", nb.toS))
    cb.if_(T.shapes(0) < nb * k, cb._fatal("gemqrt: invalid T size"))
    if (side == "L") {
      cb.if_(l.cne(m), cb._fatal("gemqrt: invalid dimensions"))
      cb.if_(work.shapes(0) < nb * n, cb._fatal("work array too small"))
    } else {
      cb.if_(l.cne(n), cb._fatal("gemqrt: invalid dimensions"))
      cb.if_(work.shapes(0) < nb * m, cb._fatal("work array too small"))
    }

    val error = cb.mb.newLocal[Int]()
    val ldV = V.eltStride(1).max(1)
    val ldC = if (C.st.nDims == 2) C.eltStride(1).max(1) else m.toI
    cb.assign(
      error,
      Code.invokeScalaObject13[
        String,
        String,
        Int,
        Int,
        Int,
        Int,
        Long,
        Int,
        Long,
        Int,
        Long,
        Int,
        Long,
        Int,
      ](
        LAPACK.getClass,
        "dgemqrt",
        side,
        trans,
        m.toI,
        n.toI,
        k.toI,
        nb.toI,
        V.firstDataAddress,
        ldV,
        T.firstDataAddress,
        nb.toI.max(1),
        C.firstDataAddress,
        ldC,
        work.firstDataAddress,
      ),
    )
    cb.if_(error.cne(0), cb._fatal("LAPACK error dgemqrt. Error code = ", error.toS))
  }

  // Computes the QR factorization of A. Stores resulting factors in Q and R, overwriting A.
  def geqrt_full(
    cb: EmitCodeBuilder,
    A: SNDArrayValue,
    Q: SNDArrayValue,
    R: SNDArrayValue,
    T: SNDArrayValue,
    work: SNDArrayValue,
    blocksize: Value[Long],
  ): Unit = {
    val Seq(_, n) = A.shapes
    SNDArray.geqrt(A, T, work, blocksize, cb)
    // copy upper triangle of A0 to R
    SNDArray.copyMatrix(cb, "U", A.slice(cb, (null, n), ColonIndex), R)

    // Set Q to I
    Q.setToZero(cb)
    val i = cb.mb.newLocal[Long]("i")
    cb.for_(
      cb.assign(i, 0L),
      i < n,
      cb.assign(i, i + 1),
      Q.setElement(FastSeq(i, i), primitive(1.0), cb),
    )
    SNDArray.gemqrt("L", "N", A, T, Q, work, blocksize, cb)
  }

  def geqr_query(cb: EmitCodeBuilder, m: Value[Long], n: Value[Long], region: Value[Region])
    : (Value[Long], Value[Long]) = {
    val T = cb.memoize(region.allocate(8L * 5, 8L))
    val work = cb.memoize(region.allocate(8L, 8L))
    val info = cb.memoize(Code.invokeScalaObject8[Int, Int, Long, Int, Long, Int, Long, Int, Int](
      LAPACK.getClass,
      "dgeqr",
      m.toI,
      n.toI,
      0,
      m.toI,
      T,
      -1,
      work,
      -1,
    ))
    cb.if_(
      info.cne(0),
      cb._fatal(s"LAPACK error DGEQR. Failed size query. Error code = ", info.toS),
    )
    val Tsize = cb.memoize(Region.loadDouble(T).toL)
    val LWork = cb.memoize(Region.loadDouble(work).toL)
    (cb.memoize(Tsize.max(5)), cb.memoize(LWork.max(1)))
  }

  def geqr(cb: EmitCodeBuilder, A: SNDArrayValue, T: SNDArrayValue, work: SNDArrayValue): Unit = {
    assertMatrix(A)
    assertColMajor(cb, "geqr", A)
    assertVector(T, work)

    val Seq(m, n) = A.shapes
    val lwork = work.shapes(0)
    val Tsize = T.shapes(0)

    val ldA = A.eltStride(1).max(1)
    val info = cb.newLocal[Int]("dgeqrf_info")
    cb.assign(
      info,
      Code.invokeScalaObject8[Int, Int, Long, Int, Long, Int, Long, Int, Int](
        LAPACK.getClass,
        "dgeqr",
        m.toI,
        n.toI,
        A.firstDataAddress,
        ldA,
        T.firstDataAddress,
        Tsize.toI,
        work.firstDataAddress,
        lwork.toI,
      ),
    )
    val optTsize = T.loadElement(FastSeq(0), cb).asFloat64.value.toI
    val optLwork = work.loadElement(FastSeq(0), cb).asFloat64.value.toI
    cb.if_(optTsize > Tsize.toI, cb._fatal(s"dgeqr: T too small"))
    cb.if_(optLwork > lwork.toI, cb._fatal(s"dgeqr: work too small"))
    cb.if_(info.cne(0), cb._fatal(s"LAPACK error dgeqr. Error code = ", info.toS))
  }

  def gemqr(
    cb: EmitCodeBuilder,
    side: String,
    trans: String,
    A: SNDArrayValue,
    T: SNDArrayValue,
    C: SNDArrayValue,
    work: SNDArrayValue,
  ): Unit = {
    assertMatrix(A)
    assertColMajor(cb, "gemqr", A)
    if (C.st.nDims == 2) assertColMajor(cb, "gemqr", C) else assertVector(C)
    assertVector(work, T)

    assert(side == "L" || side == "R")
    assert(trans == "T" || trans == "N")
    val Seq(l, k) = A.shapes
    val Seq(m, n) = if (C.st.nDims == 2) C.shapes else FastSeq(C.shapes(0), SizeValueStatic(1))
    if (side == "L") {
      cb.if_(l.cne(m), cb._fatal("gemqr: invalid dimensions"))
    } else {
      cb.if_(l.cne(n), cb._fatal("gemqr: invalid dimensions"))
    }
    val Tsize = T.shapes(0)
    val Lwork = work.shapes(0)

    val error = cb.mb.newLocal[Int]()
    val ldA = A.eltStride(1).max(1)
    val ldC = if (C.st.nDims == 2) C.eltStride(1).max(1) else m.toI
    cb.assign(
      error,
      Code.invokeScalaObject13[
        String,
        String,
        Int,
        Int,
        Int,
        Long,
        Int,
        Long,
        Int,
        Long,
        Int,
        Long,
        Int,
        Int,
      ](
        LAPACK.getClass,
        "dgemqr",
        side,
        trans,
        m.toI,
        n.toI,
        k.toI,
        A.firstDataAddress,
        ldA,
        T.firstDataAddress,
        Tsize.toI,
        C.firstDataAddress,
        ldC,
        work.firstDataAddress,
        Lwork.toI,
      ),
    )
    cb.if_(error.cne(0), cb._fatal("LAPACK error dgemqr. Error code = ", error.toS))
  }

  // Computes the QR factorization of A. Stores resulting factors in Q and R, overwriting A.
  def geqr_full(
    cb: EmitCodeBuilder,
    A: SNDArrayValue,
    Q: SNDArrayValue,
    R: SNDArrayValue,
    T: SNDArrayValue,
    work: SNDArrayValue,
  ): Unit = {
    val Seq(_, n) = A.shapes
    SNDArray.geqr(cb, A, T, work)
    // copy upper triangle of A0 to R
    SNDArray.copyMatrix(cb, "U", A.slice(cb, (null, n), ColonIndex), R)

    // Set Q to I
    Q.setToZero(cb)
    val i = cb.mb.newLocal[Long]("i")
    cb.for_(
      cb.assign(i, 0L),
      i < n,
      cb.assign(i, i + 1),
      Q.setElement(FastSeq(i, i), primitive(1.0), cb),
    )
    SNDArray.gemqr(cb, "L", "N", A, T, Q, work)
  }

  def tpqrt(
    A: SNDArrayValue,
    B: SNDArrayValue,
    T: SNDArrayValue,
    work: SNDArrayValue,
    blocksize: Value[Long],
    cb: EmitCodeBuilder,
  ): Unit = {
    assertMatrix(A, B)
    assertColMajor(cb, "tpqrt", A, B)
    assertVector(work, T)

    val Seq(m, n) = B.shapes
    val nb = blocksize
    cb.if_(nb > n || nb < 1, cb._fatal("tpqrt: invalid block size"))
    cb.if_(T.shapes(0) < nb * n, cb._fatal("tpqrt: T too small"))
    A.assertHasShape(cb, FastSeq(n, n), "tpqrt: invalid shapes")
    cb.if_(work.shapes(0) < nb * n, cb._fatal("tpqrt: work array too small"))

    val error = cb.mb.newLocal[Int]()
    val ldA = A.eltStride(1).max(1)
    val ldB = B.eltStride(1).max(1)
    cb.assign(
      error,
      Code.invokeScalaObject11[Int, Int, Int, Int, Long, Int, Long, Int, Long, Int, Long, Int](
        LAPACK.getClass,
        "dtpqrt",
        m.toI,
        n.toI,
        0,
        nb.toI,
        A.firstDataAddress,
        ldA,
        B.firstDataAddress,
        ldB,
        T.firstDataAddress,
        nb.toI.max(1),
        work.firstDataAddress,
      ),
    )
    cb.if_(error.cne(0), cb._fatal("LAPACK error dtpqrt. Error code = ", error.toS))
  }

  def tpmqrt(
    side: String,
    trans: String,
    V: SNDArrayValue,
    T: SNDArrayValue,
    A: SNDArrayValue,
    B: SNDArrayValue,
    work: SNDArrayValue,
    blocksize: Value[Long],
    cb: EmitCodeBuilder,
  ): Unit = {
    assertMatrix(A, B, V)
    assertColMajor(cb, "tpmqrt", A, B, V)
    assertVector(work, T)

    assert(side == "L" || side == "R")
    assert(trans == "T" || trans == "N")
    val Seq(l, k) = V.shapes
    val Seq(m, n) = B.shapes
    val nb = blocksize
    cb.if_(nb > k || nb < 1, cb._fatal("tpmqrt: invalid block size"))
    cb.if_(T.shapes(0) < nb * k, cb._fatal("tpmqrt: T too small"))
    if (side == "L") {
      cb.if_(l.cne(m), cb._fatal("tpmqrt: invalid dimensions"))
      cb.if_(work.shapes(0) < nb * n, cb._fatal("tpmqrt: work array too small"))
      A.assertHasShape(cb, FastSeq(k, n), "tpmqrt: invalid shapes")
    } else {
      cb.if_(l.cne(n), cb._fatal("invalid dimensions"))
      cb.if_(work.shapes(0) < nb * m, cb._fatal("work array too small"))
      A.assertHasShape(cb, FastSeq(m, k), "tpmqrt: invalid shapes")
    }

    val error = cb.mb.newLocal[Int]()
    val ldV = V.eltStride(1).max(1)
    val ldA = A.eltStride(1).max(1)
    val ldB = B.eltStride(1).max(1)
    cb.assign(
      error,
      Code.invokeScalaObject16[
        String,
        String,
        Int,
        Int,
        Int,
        Int,
        Int,
        Long,
        Int,
        Long,
        Int,
        Long,
        Int,
        Long,
        Int,
        Long,
        Int,
      ](
        LAPACK.getClass,
        "dtpmqrt",
        side,
        trans,
        m.toI,
        n.toI,
        k.toI,
        0,
        nb.toI,
        V.firstDataAddress,
        ldV,
        T.firstDataAddress,
        nb.toI.max(1),
        A.firstDataAddress,
        ldA,
        B.firstDataAddress,
        ldB,
        work.firstDataAddress,
      ),
    )
    cb.if_(error.cne(0), cb._fatal("LAPACK error dtpqrt. Error code = ", error.toS))
  }

  def geqrf_query(cb: EmitCodeBuilder, m: Value[Int], n: Value[Int], region: Value[Region])
    : Value[Int] = {
    val LWorkAddress = cb.newLocal[Long]("dgeqrf_lwork_address")
    val LWork = cb.newLocal[Int]("dgeqrf_lwork")
    val info = cb.newLocal[Int]("dgeqrf_info")
    cb.assign(LWorkAddress, region.allocate(8L, 8L))
    cb.assign(
      info,
      Code.invokeScalaObject7[Int, Int, Long, Int, Long, Long, Int, Int](
        LAPACK.getClass,
        "dgeqrf",
        m.toI,
        n.toI,
        0,
        m.toI,
        0,
        LWorkAddress,
        -1,
      ),
    )
    cb.if_(
      info.cne(0),
      cb._fatal(s"LAPACK error DGEQRF. Failed size query. Error code = ", info.toS),
    )
    cb.assign(LWork, Region.loadDouble(LWorkAddress).toI)
    cb.memoize((LWork > 0).mux(LWork, 1))
  }

  def geqrf(cb: EmitCodeBuilder, A: SNDArrayValue, T: SNDArrayValue, work: SNDArrayValue): Unit = {
    assertMatrix(A)
    assertColMajor(cb, "geqrf", A)
    assertVector(T, work)

    val Seq(m, n) = A.shapes
    cb.if_(T.shapes(0).cne(m.min(n)), cb._fatal("geqrf: T has wrong size"))
    val lwork = work.shapes(0)
    cb.if_(lwork < n.max(1L), cb._fatal("geqrf: work has wrong size"))

    val ldA = A.eltStride(1).max(1)
    val info = cb.newLocal[Int]("dgeqrf_info")
    cb.assign(
      info,
      Code.invokeScalaObject7[Int, Int, Long, Int, Long, Long, Int, Int](
        LAPACK.getClass,
        "dgeqrf",
        m.toI,
        n.toI,
        A.firstDataAddress,
        ldA,
        T.firstDataAddress,
        work.firstDataAddress,
        lwork.toI,
      ),
    )
    cb.if_(info.cne(0), cb._fatal(s"LAPACK error DGEQRF. Error code = ", info.toS))
  }

  def orgqr(
    cb: EmitCodeBuilder,
    k: Value[Int],
    A: SNDArrayValue,
    T: SNDArrayValue,
    work: SNDArrayValue,
  ): Unit = {
    assertMatrix(A)
    assertColMajor(cb, "orgqr", A)
    assertVector(T, work)

    val Seq(m, n) = A.shapes
    cb.if_(k < 0 || k > n.toI, cb._fatal("orgqr: invalid k"))
    cb.if_(T.shapes(0).cne(m.min(n)), cb._fatal("orgqr: T has wrong size"))
    val lwork = work.shapes(0)
    cb.if_(lwork < n.max(1L), cb._fatal("orgqr: work has wrong size"))

    val ldA = A.eltStride(1).max(1)
    val info = cb.newLocal[Int]("dgeqrf_info")
    cb.assign(
      info,
      Code.invokeScalaObject8[Int, Int, Int, Long, Int, Long, Long, Int, Int](
        LAPACK.getClass,
        "dorgqr",
        m.toI,
        n.toI,
        k.toI,
        A.firstDataAddress,
        ldA,
        T.firstDataAddress,
        work.firstDataAddress,
        lwork.toI,
      ),
    )
    cb.if_(info.cne(0), cb._fatal(s"LAPACK error DGEQRF. Error code = ", info.toS))
  }

  def syevr_query(
    cb: EmitCodeBuilder,
    jobz: String,
    uplo: String,
    n: Value[Int],
    region: Value[Region],
  ): (SizeValue, SizeValue) = {
    val WorkAddress = cb.memoize(region.allocate(8L, 8L))
    val IWorkAddress = cb.memoize(region.allocate(4L, 4L))
    val info = cb.memoize(Code.invokeScalaObject19[
      String,
      String,
      String,
      Int,
      Long,
      Int,
      Double,
      Double,
      Int,
      Int,
      Double,
      Long,
      Long,
      Int,
      Long,
      Long,
      Int,
      Long,
      Int,
      Int,
    ](
      LAPACK.getClass,
      "dsyevr",
      jobz,
      "A",
      uplo,
      n,
      0,
      n,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      n,
      0,
      WorkAddress,
      -1,
      IWorkAddress,
      -1,
    ))
    cb.if_(
      info.cne(0),
      cb._fatal(s"LAPACK error DSYEVR. Failed size query. Error code = ", info.toS),
    )
    val LWork = cb.memoize(Region.loadDouble(WorkAddress).toL)
    val LIWork = cb.memoize(Region.loadInt(IWorkAddress).toL)
    (
      SizeValueDyn(cb.memoize((LWork > 0).mux(LWork, 1))),
      SizeValueDyn(cb.memoize((LIWork > 0).mux(LIWork, 1))),
    )
  }

  def syevr(
    cb: EmitCodeBuilder,
    uplo: String,
    A: SNDArrayValue,
    W: SNDArrayValue,
    Z: Option[(SNDArrayValue, SNDArrayValue)],
    Work: SNDArrayValue,
    IWork: SNDArrayValue,
  ): Unit = {
    assertMatrix(A)
    assertColMajor(cb, "orgqr", A)
    assertVector(W, Work, IWork)
    assert(IWork.pt.elementType.virtualType == TInt32)

    val n = A.shapes(0)
    A.assertHasShape(cb, Array(n, n), "syevr: A must be square")
    W.assertHasShape(cb, Array(n), "syevr: W has wrong size")

    val ldA = A.eltStride(1).max(1)
    val lWork = Work.shapes(0)
    val lIWork = IWork.shapes(0)

    val (jobz, zAddr: Value[Long], ldZ: Code[Int], iSuppZAddr: Value[Long]) = Z match {
      case Some((z, iSuppZ)) =>
        assertVector(iSuppZ)
        assertMatrix(z)

        z.assertHasShape(cb, Array(n, n), "syevr: Z has wrong size")
        iSuppZ.assertHasShape(
          cb,
          IndexedSeq(SizeValueDyn(cb.memoize(n * 2))),
          "syevr: ISuppZ has wrong size",
        )

        ("V", z.firstDataAddress, z.eltStride(1).max(1), iSuppZ.firstDataAddress)
      case None =>
        ("N", const(0L), const(1).get, const(0L))
    }

    val info = cb.memoize(Code.invokeScalaObject19[
      String,
      String,
      String,
      Int,
      Long,
      Int,
      Double,
      Double,
      Int,
      Int,
      Double,
      Long,
      Long,
      Int,
      Long,
      Long,
      Int,
      Long,
      Int,
      Int,
    ](
      LAPACK.getClass,
      "dsyevr",
      jobz,
      "A",
      uplo,
      n.toI,
      A.firstDataAddress,
      ldA,
      0,
      0,
      0,
      0,
      0,
      W.firstDataAddress,
      zAddr,
      ldZ,
      iSuppZAddr,
      Work.firstDataAddress,
      lWork.toI,
      IWork.firstDataAddress,
      lIWork.toI,
    ))
    cb.if_(info.cne(0), cb._fatal(s"LAPACK error DSYEVR. Error code = ", info.toS))
  }
}

trait SNDArray extends SType {
  def pType: PNDArray

  def nDims: Int

  def elementType: SType
  def elementPType: PType
  def elementEmitType: EmitType = EmitType(elementType, pType.elementType.required)

  def elementByteSize: Long

  override def _typeWithRequiredness: TypeWithRequiredness =
    RNDArray(elementType.typeWithRequiredness.setRequired(true).r)
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

  def pt: PNDArray

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
      cb.if_(
        indices(dimIndex) >= shape(dimIndex) || indices(dimIndex) < 0,
        cb._fatalWithError(
          errorId,
          "Index ",
          indices(dimIndex).toS,
          s" is out of bounds for axis $dimIndex with size ",
          shape(dimIndex).toS,
        ),
      )
    }
  }

  def sameShape(cb: EmitCodeBuilder, other: SNDArrayValue): Code[Boolean] =
    hasShape(cb, other.shapes)

  def hasShape(cb: EmitCodeBuilder, otherShape: IndexedSeq[SizeValue]): Code[Boolean] = {
    var b: Code[Boolean] = const(true)
    val shape = this.shapes
    assert(shape.length == otherShape.length)

    (shape, otherShape).zipped.foreach((s1, s2) => b = s1.ceq(s2))
    b
  }

  def assertHasShape(cb: EmitCodeBuilder, otherShape: IndexedSeq[SizeValue], msg: Code[String]*) =
    if (!hasShapeStatic(otherShape))
      cb.if_(
        !hasShape(cb, otherShape),
        cb._fatal(
          msg ++
            (const("\nExpected shape ").get +:
              shapes.map(_.toS).intersperse[Code[String]]("(", ",", ")")) ++
            (const(", found ").get +:
              otherShape.map(_.toS).intersperse[Code[String]]("(", ",", ")")): _*
        ),
      )

  // True IFF shape can be proven equal to otherShape statically
  def hasShapeStatic(otherShape: IndexedSeq[SizeValue]): Boolean =
    shapes == otherShape

  def hasShapeStatic(otherShape: SizeValue*): Boolean =
    hasShapeStatic(otherShape.toFastSeq)

  def isVector: Boolean = shapes.length == 1

  // ensure coerceToShape(cb, otherShape).hasShapeStatic(otherShape)
  // Inserts any necessary dynamic assertions
  def coerceToShape(cb: EmitCodeBuilder, otherShape: IndexedSeq[SizeValue]): SNDArrayValue

  def coerceToShape(cb: EmitCodeBuilder, otherShape: SizeValue*): SNDArrayValue =
    coerceToShape(cb, otherShape.toFastSeq)

  def contiguousDimensions(cb: EmitCodeBuilder): Value[Int] = {
    val tmp = cb.mb.newLocal[Int]("NDArray_setToZero_tmp")
    val contiguousDims = cb.mb.newLocal[Int]("NDArray_setToZero_contigDims")

    cb.assign(tmp, 1)

    // Find largest prefix of dimensions which are stored contiguously.
    def contigDimsRecur(i: Int): Unit =
      if (i < st.nDims) {
        cb.if_(
          tmp.ceq(eltStride(i)), {
            cb.assign(tmp, tmp * shapes(i).toI)
            contigDimsRecur(i + 1)
          },
          cb.assign(contiguousDims, i),
        )
      } else {
        cb.assign(contiguousDims, st.nDims)
      }

    contigDimsRecur(0)

    contiguousDims
  }

  // FIXME: only optimized for column major
  def setToZero(cb: EmitCodeBuilder): Unit = {
    val eltType = pt.elementType.asInstanceOf[PNumeric with PPrimitive]

    val contiguousDims = contiguousDimensions(cb)

    def recur(startPtr: Value[Long], dim: Int, contiguousDims: Int): Unit =
      if (dim > 0) {
        if (contiguousDims == dim)
          cb += Region.setMemory(startPtr, shapes(dim - 1) * strides(dim - 1), 0: Byte)
        else {
          val ptr = cb.mb.newLocal[Long](s"NDArray_setToZero_ptr_$dim")
          val end = cb.mb.newLocal[Long](s"NDArray_setToZero_end_$dim")
          cb.assign(ptr, startPtr)
          cb.assign(end, startPtr + strides(dim - 1) * shapes(dim - 1))
          cb.for_(
            {},
            ptr < end,
            cb.assign(ptr, ptr + strides(dim - 1)),
            recur(ptr, dim - 1, contiguousDims),
          )
        }
      } else {
        eltType.storePrimitiveAtAddress(cb, startPtr, primitive(eltType.virtualType, eltType.zero))
      }

    cb.switch(
      contiguousDims,
      recur(firstDataAddress, st.nDims, 2),
      FastSeq(
        () => recur(firstDataAddress, st.nDims, 0),
        () => recur(firstDataAddress, st.nDims, 1),
      ),
    )
  }

  def setElement(indices: IndexedSeq[Value[Long]], value: SValue, cb: EmitCodeBuilder): Unit = {
    val eltType = st.pType.elementType.asInstanceOf[PPrimitive]
    eltType.storePrimitiveAtAddress(cb, loadElementAddress(indices, cb), value)
  }

  def coiterateMutate(
    cb: EmitCodeBuilder,
    region: Value[Region],
    arrays: (SNDArrayValue, String)*
  )(
    body: IndexedSeq[SValue] => SValue
  ): Unit =
    coiterateMutate(cb, region, false, arrays: _*)(body)

  def coiterateMutate(
    cb: EmitCodeBuilder,
    region: Value[Region],
    deepCopy: Boolean,
    arrays: (SNDArrayValue, String)*
  )(
    body: IndexedSeq[SValue] => SValue
  ): Unit = {
    val indexVars = Array.tabulate(st.nDims)(i => s"i$i").toFastSeq
    val indices = Array.range(0, st.nDims).toFastSeq
    coiterateMutate(
      cb,
      region,
      deepCopy,
      indexVars,
      indices,
      arrays.map { case (array, name) => (array, indices, name) }: _*
    )(body)
  }

  def coiterateMutate(
    cb: EmitCodeBuilder,
    region: Value[Region],
    indexVars: IndexedSeq[String],
    destIndices: IndexedSeq[Int],
    arrays: (SNDArrayValue, IndexedSeq[Int], String)*
  )(
    body: IndexedSeq[SValue] => SValue
  ): Unit =
    coiterateMutate(cb, region, false, indexVars, destIndices, arrays: _*)(body)

  /* Note: to iterate through an array in column major order, make sure the indices are in ascending
   * order. E.g. */
  /* A.coiterateMutate(cb, region, IndexedSeq("i", "j"), IndexedSeq((A, IndexedSeq(0, 1), "A"), (B,
   * IndexedSeq(0, 1), "B")), { */
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
  )(
    body: IndexedSeq[SValue] => SValue
  ): Unit

  def _slice(cb: EmitCodeBuilder, indices: IndexedSeq[NDArrayIndex]): SNDArraySliceValue = {
    val shapeX = shapes
    val stridesX = strides
    val shapeBuilder = mutable.ArrayBuilder.make[SizeValue]
    val stridesBuilder = mutable.ArrayBuilder.make[Value[Long]]

    for (i <- indices.indices) indices(i) match {
      case ScalarIndex(j) =>
        cb.if_(
          j < 0 || j >= shapeX(i),
          cb._fatal(
            "Scalar index out of bounds (axis ",
            i.toString,
            "): ",
            j.toS,
            " is not in [0,",
            shapeX(i).toS,
            ")",
          ),
        )
      case SliceIndex(Some(begin), Some(end)) =>
        cb.if_(begin > end, cb._fatal("Invalid slice index, ", begin.toS, " > ", end.toS))
        cb.if_(
          begin < 0 || end > shapeX(i),
          cb._fatal(
            "Slice index out of bounds: (axis ",
            i.toString,
            ") range ",
            begin.toS,
            ":",
            end.toS,
            " is not contained by [0,",
            shapeX(i).toS,
            ")",
          ),
        )
        val s = cb.newLocal[Long]("slice_size", end - begin)
        shapeBuilder += SizeValueDyn(s)
        stridesBuilder += stridesX(i)
      case SliceIndex(None, Some(end: SizeValue)) =>
        cb.if_(end > shapeX(i) || end < 0, cb._fatal("Index out of bounds"))
        shapeBuilder += end
        stridesBuilder += stridesX(i)
      case SliceIndex(None, Some(end)) =>
        cb.if_(
          end < 0,
          cb._fatal(
            "Slice end index out of bounds (axis ",
            i.toString,
            "): endpoint ",
            end.toS,
            " < 0",
          ),
        )
        cb.if_(
          end > shapeX(i),
          cb._fatal("Slice end index out of bounds: endpoint ", end.toS, " > ", shapeX(i).toS),
        )
        shapeBuilder += SizeValueDyn(end)
        stridesBuilder += stridesX(i)
      case SliceIndex(Some(begin), None) =>
        cb.if_(
          begin < 0,
          cb._fatal(
            "Slice start index out of bounds (axis ",
            i.toString,
            "): startpoint ",
            begin.toS,
            " < 0",
          ),
        )
        cb.if_(
          begin > shapeX(i),
          cb._fatal(
            "Slice start index out of bounds (axis ",
            i.toString,
            "): startpoint ",
            begin.toS,
            " > ",
            shapeX(i).toS,
          ),
        )
        val s = cb.newLocal[Long]("slice_size", shapeX(i) - begin)
        shapeBuilder += SizeValueDyn(s)
        stridesBuilder += stridesX(i)
      case SliceIndex(None, None) =>
        shapeBuilder += shapeX(i)
        stridesBuilder += stridesX(i)
      case SliceSize(None, size) =>
        cb.if_(
          size > shapeX(i),
          cb._fatal(
            "Slice size out of bounds (axis ",
            i.toString,
            "): size ",
            size.toS,
            " > ",
            shapeX(i).toS,
          ),
        )
        shapeBuilder += size
        stridesBuilder += stridesX(i)
      case SliceSize(Some(begin), size) =>
        cb.if_(
          begin < 0,
          cb._fatal("Slice start out of bounds (axis ", i.toString, "): start ", begin.toS, " < 0"),
        )
        cb.if_(
          begin + size > shapeX(i),
          cb._fatal(
            "Slice index out of bounds (axis ",
            i.toString,
            "): range ",
            begin.toS,
            ":",
            begin.toS,
            "+",
            size.toS,
            " is not contained by [0,",
            shapeX(i).toS,
            ")",
          ),
        )
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

    val newFirstDataAddress =
      cb.newLocal[Long]("slice_ptr", loadElementAddress(firstElementIndices, cb))

    val newSType =
      SNDArraySlice(PCanonicalNDArray(st.pType.elementType, newShape.size, st.pType.required))

    new SNDArraySliceValue(newSType, newShape, newStrides, newFirstDataAddress)
  }

  def slice(cb: EmitCodeBuilder, indices: Any*): SNDArraySliceValue = {
    val parsedIndices: IndexedSeq[NDArrayIndex] = indices.map {
      case ColonIndex => ColonIndex
      case i: Value[_] => ScalarIndex(i.asInstanceOf[Value[Long]])
      case i: Code[_] => ScalarIndex(cb.memoize(coerce[Long](i)))
      case i: Int => ScalarIndex(const(i.toLong))
      case i: Long => ScalarIndex(const(i))
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

  override def sizeToStoreInBytes(cb: EmitCodeBuilder): SInt64Value = {
    val storageType = st.storageType().asInstanceOf[PNDArray]
    val totalSize = cb.newLocal[Long]("sindexableptr_size_in_bytes", storageType.byteSize)

    if (storageType.elementType.containsPointers) {
      SNDArray.coiterate(cb, (this, "A")) {
        case Seq(elt) =>
          cb.assign(totalSize, totalSize + elt.sizeToStoreInBytes(cb).value)
      }
    } else {
      val numElements = SNDArray.numElements(this.shapes)
      cb.assign(totalSize, totalSize + (numElements * storageType.elementType.byteSize))
    }
    new SInt64Value(totalSize)
  }
}

trait SNDArraySettable extends SNDArrayValue with SSettable

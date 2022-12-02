package is.hail.types.physical.stypes.interfaces

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.linalg.{BLAS, LAPACK}
import is.hail.types.physical.stypes.concrete.{SNDArraySlice, SNDArraySliceValue}
import is.hail.types.physical.stypes.primitives.SInt64Value
import is.hail.types.physical.stypes.{EmitType, SSettable, SType, SValue}
import is.hail.types.physical.{PCanonicalNDArray, PNDArray, PType}
import is.hail.types.{RNDArray, TypeWithRequiredness}
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
  // A.coiterate(cb, region, IndexedSeq("i", "j"), IndexedSeq((A, IndexedSeq(0, 1), "A"), (B, IndexedSeq(0, 1), "B")), {
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
          cb.ifx(indexSizes(idx).cne(shape(i).toI), s"${indexVars(idx)} indexes incompatible dimensions")
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

  def scale(cb: EmitCodeBuilder, alpha: SValue, X: SNDArrayValue): Unit =
    scale(cb, alpha.asFloat64.value, X)

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
    gemm(cb, tA, tB, 1.0, A, B, 1.0, C)

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
    assertMatrix(A)
    assertColMajor(cb, A)
    assertVector(work, T)

    val Seq(m, n) = A.shapes
    val nb = blocksize
    cb.ifx(nb > m.min(n) || nb < 1, cb._fatal("invalid block size"))
    cb.ifx(T.shapes(0) < nb*(m.min(n)), cb._fatal("T too small"))
    cb.ifx(work.shapes(0) < nb * n, cb._fatal("work array too small"))

    val error = cb.mb.newLocal[Int]()
    val ldA = A.eltStride(1).max(1)
    cb.assign(error, Code.invokeScalaObject8[Int, Int, Int, Long, Int, Long, Int, Long, Int](LAPACK.getClass, "dgeqrt",
      m.toI, n.toI, nb.toI,
      A.firstDataAddress, ldA,
      T.firstDataAddress, nb.toI.max(1),
      work.firstDataAddress))
    cb.ifx(error.cne(0), cb._fatal("LAPACK error dtpqrt. Error code = ", error.toS))
  }

  def gemqrt(side: String, trans: String, V: SNDArrayValue, T: SNDArrayValue, C: SNDArrayValue, work: SNDArrayValue, blocksize: Value[Long], cb: EmitCodeBuilder): Unit = {
    assertMatrix(C, V)
    assertColMajor(cb, C, V)
    assertVector(work, T)

    assert(side == "L" || side == "R")
    assert(trans == "T" || trans == "N")
    val Seq(l, k) = V.shapes
    val Seq(m, n) = C.shapes
    val nb = blocksize
    cb.ifx(nb > k || nb < 1, cb._fatal("invalid block size"))
    cb.ifx(T.shapes(0) < nb*k, cb._fatal("invalid T size"))
    if (side == "L") {
      cb.ifx(l.cne(m), cb._fatal("invalid dimensions"))
      cb.ifx(work.shapes(0) < nb * n, cb._fatal("work array too small"))
    } else {
      cb.ifx(l.cne(n), cb._fatal("invalid dimensions"))
      cb.ifx(work.shapes(0) < nb * m, cb._fatal("work array too small"))
    }

    val error = cb.mb.newLocal[Int]()
    val ldV = V.eltStride(1).max(1)
    val ldC = C.eltStride(1).max(1)
    cb.assign(error, Code.invokeScalaObject13[String, String, Int, Int, Int, Int, Long, Int, Long, Int, Long, Int, Long, Int](LAPACK.getClass, "dgemqrt",
      side, trans, m.toI, n.toI, k.toI, nb.toI,
      V.firstDataAddress, ldV,
      T.firstDataAddress, nb.toI.max(1),
      C.firstDataAddress, ldC,
      work.firstDataAddress))
    cb.ifx(error.cne(0), cb._fatal("LAPACK error dtpqrt. Error code = ", error.toS))
  }

  def tpqrt(A: SNDArrayValue, B: SNDArrayValue, T: SNDArrayValue, work: SNDArrayValue, blocksize: Value[Long], cb: EmitCodeBuilder): Unit = {
    assertMatrix(A, B)
    assertColMajor(cb, A, B)
    assertVector(work, T)

    val Seq(m, n) = B.shapes
    val nb = blocksize
    cb.ifx(nb > n || nb < 1, cb._fatal("invalid block size"))
    cb.ifx(T.shapes(0) < nb*n, cb._fatal("T too small"))
    A.assertHasShape(cb, FastIndexedSeq(n, n))
    cb.ifx(work.shapes(0) < nb * n, cb._fatal("work array too small"))

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
    cb.ifx(nb > k || nb < 1, cb._fatal("invalid block size"))
    cb.ifx(T.shapes(0) < nb*k, cb._fatal("T too small"))
    if (side == "L") {
      cb.ifx(l.cne(m), cb._fatal("invalid dimensions"))
      cb.ifx(work.shapes(0) < nb * n, cb._fatal("work array too small"))
      A.assertHasShape(cb, FastIndexedSeq(k, n))
    } else {
      cb.ifx(l.cne(n), cb._fatal("invalid dimensions"))
      cb.ifx(work.shapes(0) < nb * m, cb._fatal("work array too small"))
      A.assertHasShape(cb, FastIndexedSeq(m, k))
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

  // ensure coerceToShape(cb, otherShape).hasShapeStatic(otherShape)
  // Inserts any necessary dynamic assertions
  def coerceToShape(cb: EmitCodeBuilder, otherShape: IndexedSeq[SizeValue]): SNDArrayValue

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
  // A.coiterate(cb, region, IndexedSeq("i", "j"), IndexedSeq((A, IndexedSeq(0, 1), "A"), (B, IndexedSeq(0, 1), "B")), {
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

  def slice(cb: EmitCodeBuilder, indices: IndexedSeq[NDArrayIndex]): SNDArraySliceValue = {
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
      case SliceIndex(None, Some(end)) =>
        cb.ifx(end >= shapeX(i) || end < 0, cb._fatal("Index out of bounds"))
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

  override def sizeToStoreInBytes(cb: EmitCodeBuilder): SInt64Value = {
    val storageType = st.storageType().asInstanceOf[PNDArray]
    val totalSize = cb.newLocal[Long]("sindexableptr_size_in_bytes", storageType.byteSize)

    if (storageType.elementType.containsPointers) {
      SNDArray.coiterate(cb, (this, "A")){
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

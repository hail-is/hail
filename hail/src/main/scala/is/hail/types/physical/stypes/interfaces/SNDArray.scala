package is.hail.types.physical.stypes.interfaces

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.{RNDArray, TypeWithRequiredness}
import is.hail.types.physical.{PNDArray, PCanonicalNDArray, PType}
import is.hail.types.physical.stypes.concrete.{SNDArraySlice, SNDArraySliceCode}
import is.hail.linalg.{BLAS, LAPACK}
import is.hail.types.physical.stypes.{SCode, SSettable, SType, SValue}
import is.hail.utils.toRichIterable

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

  def coiterate(cb: EmitCodeBuilder, arrays: (SNDArrayCode, String)*)(body: IndexedSeq[SCode] => Unit): Unit = {
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
    arrays: (SNDArrayCode, IndexedSeq[Int], String)*
  )(body: IndexedSeq[SCode] => Unit
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
    arrays: (SNDArrayCode, IndexedSeq[Int], String)*
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

    val info = arrays.toIndexedSeq.map { case (_array, indices, name) =>
      for (idx <- indices) assert(idx < indexVars.length && idx >= 0)
      // FIXME: relax this assumption to handle transposing, non-column major
      for (i <- 0 until indices.length - 1) assert(indices(i) < indices(i+1))
      assert(indices.length == _array.st.nDims)

      val array = _array.memoize(cb, s"${name}_copy")
      val shape = array.shapes(cb)
      for (i <- indices.indices) {
        val idx = indices(i)
        if (indexSizes(idx) == null) {
          indexSizes(idx) = cb.newLocal[Int](s"${indexVars(idx)}_max")
          cb.assign(indexSizes(idx), shape(i).toI)
        } else {
          cb.ifx(indexSizes(idx).cne(shape(i).toI), s"${indexVars(idx)} indexes incompatible dimensions")
        }
      }
      val strides = array.strides(cb)
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
      cb.assign(info(n).pos(info(n).array.st.nDims), info(n).array.firstDataAddress(cb))
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
      cb.ifx(nd.strides(cb)(0).cne(nd.st.pType.elementType.byteSize), cb._fatal("Require column major: found row stride ", nd.strides(cb)(0).toS, ", expected ", nd.st.pType.elementType.byteSize.toString))
    }
  }

  def gemm(cb: EmitCodeBuilder, tA: String, tB: String, A: SNDArrayCode, B: SNDArrayCode, C: SNDArrayCode): Unit =
    gemm(cb, tA, tB, 1.0, A, B, 1.0, C)

  def gemm(cb: EmitCodeBuilder, tA: String, tB: String, alpha: Code[Double], _A: SNDArrayCode, _B: SNDArrayCode, beta: Code[Double], _C: SNDArrayCode): Unit = {
    val A = _A.memoize(cb, "copy_A")
    val B = _B.memoize(cb, "copy_B")
    val C = _C.memoize(cb, "copy_C")
    assertMatrix(A, B, C)
    assertColMajor(cb, A, B, C)

    val Seq(a0, a1) = A.shapes(cb)
    val (m, ka) = if (tA == "N") (a0, a1) else (a1, a0)
    val Seq(b0, b1) = B.shapes(cb)
    val (kb, n) = if (tB == "N") (b0, b1) else (b1, b0)
    val Seq(c0, c1) = C.shapes(cb)
    cb.ifx(ka.cne(kb) || c0.cne(m) || c1.cne(n), cb._fatal("gemm: incompatible matrix dimensions"))

    val ldA = (A.strides(cb)(1).toI >> 3).max(1)
    val ldB = (B.strides(cb)(1).toI >> 3).max(1)
    val ldC = (C.strides(cb)(1).toI >> 3).max(1)
    cb += Code.invokeScalaObject13[String, String, Int, Int, Int, Double, Long, Int, Long, Int, Double, Long, Int, Unit](BLAS.getClass, "dgemm",
      tA, tB, m.toI, n.toI, ka.toI,
      alpha,
      A.firstDataAddress(cb), ldA,
      B.firstDataAddress(cb), ldB,
      beta,
      C.firstDataAddress(cb), ldC)
  }
}


trait SNDArray extends SType {
  def pType: PNDArray

  def nDims: Int

  def elementType: SType
  def elementPType: PType

  def elementByteSize: Long

  override def _typeWithRequiredness: TypeWithRequiredness = RNDArray(elementType.typeWithRequiredness.setRequired(true).r)
}

sealed abstract class NDArrayIndex
case class ScalarIndex(i: Value[Long]) extends NDArrayIndex
case class SliceIndex(begin: Option[Value[Long]], end: Option[Value[Long]]) extends NDArrayIndex
case object ColonIndex extends NDArrayIndex

trait SNDArrayValue extends SValue {
  def st: SNDArray

  override def get: SNDArrayCode

  def loadElement(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): SCode

  def loadElementAddress(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): Code[Long]

  def shapes(cb: EmitCodeBuilder): IndexedSeq[Value[Long]]

  def strides(cb: EmitCodeBuilder): IndexedSeq[Value[Long]]

  def firstDataAddress(cb: EmitCodeBuilder): Value[Long]

  def outOfBounds(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): Code[Boolean] = {
    val shape = this.shapes(cb)
    val outOfBounds = cb.newLocal[Boolean]("sndarray_out_of_bounds", false)

    (0 until st.nDims).foreach { dimIndex =>
      cb.assign(outOfBounds, outOfBounds || (indices(dimIndex) >= shape(dimIndex)))
    }
    outOfBounds
  }

  def assertInBounds(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder, errorId: Int): Unit = {
    val shape = this.shapes(cb)
    for (dimIndex <- 0 until st.nDims) {
      cb.ifx(indices(dimIndex) >= shape(dimIndex), {
        cb._fatalWithError(errorId,
          "Index ", indices(dimIndex).toS,
          s" is out of bounds for axis $dimIndex with size ",
          shape(dimIndex).toS)
      })
    }
  }


  def coiterateMutate(cb: EmitCodeBuilder, region: Value[Region], arrays: (SNDArrayCode, String)*)(body: IndexedSeq[SCode] => SCode): Unit =
    coiterateMutate(cb, region, false, arrays: _*)(body)

  def coiterateMutate(cb: EmitCodeBuilder, region: Value[Region], deepCopy: Boolean, arrays: (SNDArrayCode, String)*)(body: IndexedSeq[SCode] => SCode): Unit = {
    if (arrays.isEmpty) return
    val indexVars = Array.tabulate(arrays(0)._1.st.nDims)(i => s"i$i").toFastIndexedSeq
    val indices = Array.range(0, arrays(0)._1.st.nDims).toFastIndexedSeq
    coiterateMutate(cb, region, deepCopy, indexVars, indices, arrays.map { case (array, name) => (array, indices, name) }: _*)(body)
  }

  def coiterateMutate(cb: EmitCodeBuilder, region: Value[Region], indexVars: IndexedSeq[String], destIndices: IndexedSeq[Int], arrays: (SNDArrayCode, IndexedSeq[Int], String)*)(body: IndexedSeq[SCode] => SCode): Unit =
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
    arrays: (SNDArrayCode, IndexedSeq[Int], String)*
  )(body: IndexedSeq[SCode] => SCode
  ): Unit

  def sameShape(other: SNDArrayValue, cb: EmitCodeBuilder): Code[Boolean] = {
    val otherShapes = other.shapes(cb)
    val b = cb.newLocal[Boolean]("sameShape_b", true)
    val shape = this.shapes(cb)
    assert(shape.length == otherShapes.length)
    shape.zip(otherShapes).foreach { case (s1, s2) =>
      cb.assign(b, b && s1.ceq(s2))
    }
    b
  }

  def slice(cb: EmitCodeBuilder, indices: IndexedSeq[NDArrayIndex]): SNDArraySliceCode = {
    val shapeX = shapes(cb)
    val stridesX = strides(cb)
    val shapeBuilder = mutable.ArrayBuilder.make[Code[Long]]
    val stridesBuilder = mutable.ArrayBuilder.make[Code[Long]]

    for (i <- indices.indices) indices(i) match {
      case ScalarIndex(j) =>
        cb.ifx(j < 0 || j >= shapeX(i), cb._fatal("Index out of bounds"))
      case SliceIndex(Some(begin), Some(end)) =>
        cb.ifx(begin < 0 || end > shapeX(i) || begin > end, cb._fatal("Index out of bounds"))
        shapeBuilder += end - begin
        stridesBuilder += stridesX(i)
      case SliceIndex(None, Some(end)) =>
        cb.ifx(end >= shapeX(i) || end < 0, cb._fatal("Index out of bounds"))
        shapeBuilder += end
        stridesBuilder += stridesX(i)
      case SliceIndex(Some(begin), None) =>
        val end = shapeX(i)
        cb.ifx(begin < 0 || begin > end, cb._fatal("Index out of bounds"))
        shapeBuilder += end - begin
        stridesBuilder += stridesX(i)
      case SliceIndex(None, None) =>
        shapeBuilder += shapeX(i)
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

    val newFirstDataAddress = loadElementAddress(firstElementIndices, cb)

    val newSType = SNDArraySlice(PCanonicalNDArray(st.pType.elementType, newShape.size, st.pType.required))

    new SNDArraySliceCode(newSType, newShape, newStrides, newFirstDataAddress)
  }
}

trait SNDArraySettable extends SNDArrayValue with SSettable

trait SNDArrayCode extends SCode {
  def st: SNDArray

  def shape(cb: EmitCodeBuilder): SBaseStructCode

  def memoize(cb: EmitCodeBuilder, name: String): SNDArrayValue
}

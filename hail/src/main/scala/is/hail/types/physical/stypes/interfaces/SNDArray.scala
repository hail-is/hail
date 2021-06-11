package is.hail.types.physical.stypes.interfaces

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.{PNDArray, PType}
import is.hail.types.physical.stypes.{SCode, SSettable, SType, SValue}
import is.hail.utils.{FastIndexedSeq, toRichIterable}

object SNDArray {
  def numElements(shape: IndexedSeq[Value[Long]]): Code[Long] = {
    shape.foldLeft(1L: Code[Long])(_ * _)
  }

  // Column major order
  def forEachIndex(cb: EmitCodeBuilder, shape: IndexedSeq[Value[Long]], context: String)
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
              cb.assign(dimVar, 0L)
            }, dimVar < shape(dimIdx), {
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

  def coiterate(cb: EmitCodeBuilder, region: Value[Region], arrays: IndexedSeq[(SNDArrayCode, String)], body: IndexedSeq[SSettable] => Unit): Unit =
    coiterate(cb, region, arrays, body, deepCopy=false)

  def coiterate(cb: EmitCodeBuilder, region: Value[Region], arrays: IndexedSeq[(SNDArrayCode, String)], body: IndexedSeq[SSettable] => Unit, deepCopy: Boolean): Unit = {
    if (arrays.isEmpty) return
    val indexVars = Array.tabulate(arrays(0)._1.st.nDims)(i => s"i$i").toFastIndexedSeq
    val indices = Array.range(0, arrays(0)._1.st.nDims).toFastIndexedSeq
    coiterate(cb, region, indexVars, arrays.map { case (array, name) => (array, indices, name) }, body, deepCopy)
  }

  def coiterate(cb: EmitCodeBuilder, region: Value[Region], indexVars: IndexedSeq[String], arrays: IndexedSeq[(SNDArrayCode, IndexedSeq[Int], String)], body: IndexedSeq[SSettable] => Unit): Unit =
    coiterate(cb, region, indexVars, arrays, body, deepCopy=false)

  // Note: to iterate through an array in column major order, make sure the indices are in ascending order. E.g.
  // coiterate(cb, region, IndexedSeq("i", "j"), IndexedSeq((A, IndexedSeq(0, 1), "A"), (B, IndexedSeq(0, 1), "B")), {
  //   case Seq(a, b) => cb.assign(a, SCode.add(cb, a, b))
  // })
  // computes A += B.
  def coiterate(cb: EmitCodeBuilder, region: Value[Region], indexVars: IndexedSeq[String], arrays: IndexedSeq[(SNDArrayCode, IndexedSeq[Int], String)], body: IndexedSeq[SSettable] => Unit, deepCopy: Boolean): Unit = {

    val indexSizes = new Array[Settable[Int]](indexVars.length)
    val indexCoords = Array.tabulate(indexVars.length) { i => cb.newLocal[Int](indexVars(i)) }

    case class ArrayInfo(
      array: SNDArrayValue,
      strides: IndexedSeq[Value[Long]],
      pos: IndexedSeq[Settable[Long]],
      elt: SSettable,
      indexToDim: Map[Int, Int])

    val info = arrays.map { case (_array, indices, name) =>
      for (idx <- indices) assert(idx < indexVars.length && idx >= 0)
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
      val pos = Array.tabulate(array.st.nDims) { i => cb.newLocal[Long](s"$name$i") }
      val elt = new SSettable {
        def st: SType = array.st.elementType
        val pt: PType = array.st.pType.elementType

        def get: SCode = pt.loadCheapPCode(cb, pt.loadFromNested(pos.last))
        def store(cb: EmitCodeBuilder, v: SCode): Unit = pt.storeAtAddress(cb, pos.last, region, v, deepCopy)
        def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(pos.last)
      }
      val indexToDim = Map(indices.indices.map(i => indices(i) -> i): _*)
      ArrayInfo(array, strides, pos, elt, indexToDim)
    }

    def recurLoopBuilder(idx: Int): Unit = {
      if (idx < 0) {
        body(info.map(_.elt))
      } else {
        val coord = indexCoords(idx)
        def init(): Unit = {
          cb.assign(coord, 0)
          for (n <- arrays.indices) {
            if (info(n).indexToDim.contains(idx)) {
              val i = info(n).indexToDim(idx)
              cb.assign(info(n).pos(i), if (i == 0) info(n).array.firstDataAddress(cb) else info(n).pos(i-1))
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

    recurLoopBuilder(indexVars.length - 1)
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
}


trait SNDArray extends SType {
  def pType: PNDArray

  def nDims: Int

  def elementType: SType
}

trait SNDArrayValue extends SValue {
  def st: SNDArray

  override def get: SNDArrayCode

  def loadElement(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): SCode

  def shapes(cb: EmitCodeBuilder): IndexedSeq[Value[Long]]

  def strides(cb: EmitCodeBuilder): IndexedSeq[Value[Long]]

  def outOfBounds(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): Code[Boolean]

  def assertInBounds(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder, errorId: Int = -1): Code[Unit]

  def sameShape(other: SNDArrayValue, cb: EmitCodeBuilder): Code[Boolean]

  def firstDataAddress(cb: EmitCodeBuilder): Value[Long]
}

trait SNDArrayCode extends SCode {
  def st: SNDArray

  def shape(cb: EmitCodeBuilder): SBaseStructCode

  def memoize(cb: EmitCodeBuilder, name: String): SNDArrayValue
}

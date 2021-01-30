package is.hail.types.physical.stypes.interfaces

import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.{SCode, SType, SValue}


object SNDArray {
  def forEachIndex(cb: EmitCodeBuilder, shape: IndexedSeq[Value[Long]], context: String)
    (f: (EmitCodeBuilder, IndexedSeq[Value[Long]]) => Unit): Unit = {

    val indices = new Array[Value[Long]](shape.length)

    def recur(dimIdx: Int): Unit = {
      if (dimIdx == indices.length) {
        f(cb, indices)
      } else {
        val currentIdx = cb.newLocal[Long](s"${ context }_foreach_dim_$dimIdx", 0L)
        indices(dimIdx) = currentIdx

        cb.whileLoop(currentIdx < shape(dimIdx),
          {
            recur(dimIdx + 1)
            cb.assign(currentIdx, currentIdx + 1L)
          })
      }
    }

    recur(0)
  }
}


trait SNDArray extends SType

trait SNDArrayValue extends SValue {
  def loadElement(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): SCode

  def shapes(cb: EmitCodeBuilder): IndexedSeq[Value[Long]]

  def strides(cb: EmitCodeBuilder): IndexedSeq[Value[Long]]

  def outOfBounds(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): Code[Boolean]

  def assertInBounds(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder, errorId: Int = -1): Code[Unit]

  def sameShape(other: SNDArrayValue, cb: EmitCodeBuilder): Code[Boolean]

  def dataAddress(cb: EmitCodeBuilder): Code[Long]
}

trait SNDArrayCode extends SCode {
  def shape(cb: EmitCodeBuilder): SBaseStructCode

  def memoize(cb: EmitCodeBuilder, name: String): SNDArrayValue
}

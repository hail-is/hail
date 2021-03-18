package is.hail.types.physical.stypes.interfaces

import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.{SCode, SType, SValue}

object SNDArray {
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
  def nDims: Int

  def elementType: SType
}

trait SNDArrayValue extends SValue {
  def st: SNDArray

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

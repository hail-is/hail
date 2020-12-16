package is.hail.types.physical.stypes.interfaces

import is.hail.asm4s.{Code, LineNumber, Value}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.{PBaseStructCode, PCode, PNDArray, PValue}
import is.hail.types.physical.stypes.{SCode, SType, SValue}

trait SNDArray extends SType

trait SNDArrayValue extends SValue {
  def loadElement(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): SCode

  def shapes(cb: EmitCodeBuilder): IndexedSeq[Value[Long]]

  def strides(cb: EmitCodeBuilder): IndexedSeq[Value[Long]]

  def outOfBounds(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): Code[Boolean]

  def assertInBounds(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder, errorId: Int = -1): Code[Unit]

  def sameShape(other: SNDArrayValue, cb: EmitCodeBuilder): Code[Boolean]
}

trait SNDArrayCode extends SCode {
  def shape(implicit line: LineNumber): SBaseStructCode

  def memoize(cb: EmitCodeBuilder, name: String): SNDArrayValue
}

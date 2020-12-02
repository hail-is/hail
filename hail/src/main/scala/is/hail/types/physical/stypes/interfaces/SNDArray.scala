package is.hail.types.physical.stypes.interfaces

import is.hail.asm4s.{Code, LineNumber, Value}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.{PBaseStructCode, PCode, PNDArray, PValue}
import is.hail.types.physical.stypes.{SCode, SType, SValue}

trait SNDArray extends SType

trait SNDArrayValue extends SValue {
  def loadElement(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder)(implicit line: LineNumber): SCode

  def shapes(cb: EmitCodeBuilder)(implicit line: LineNumber): IndexedSeq[Value[Long]]

  def strides(cb: EmitCodeBuilder)(implicit line: LineNumber): IndexedSeq[Value[Long]]

  def outOfBounds(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder)(implicit line: LineNumber): Code[Boolean]

  def assertInBounds(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder, errorId: Int = -1)(implicit line: LineNumber): Code[Unit]

  def sameShape(other: SNDArrayValue, cb: EmitCodeBuilder)(implicit line: LineNumber): Code[Boolean]
}

trait SNDArrayCode extends SCode {
  def shape(implicit line: LineNumber): SBaseStructCode

  def memoize(cb: EmitCodeBuilder, name: String)(implicit line: LineNumber): SNDArrayValue
}

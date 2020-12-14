package is.hail.types.physical.stypes.interfaces

import is.hail.asm4s.{Code, LineNumber}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.{SCode, SType, SValue}

trait SBinary extends SType

trait SBinaryValue extends SValue {
  def loadLength()(implicit line: LineNumber): Code[Int]

  def loadBytes()(implicit line: LineNumber): Code[Array[Byte]]

  def loadByte(i: Code[Int])(implicit line: LineNumber): Code[Byte]
}

trait SBinaryCode extends SCode {
  def loadLength()(implicit line: LineNumber): Code[Int]

  def loadBytes()(implicit line: LineNumber): Code[Array[Byte]]

  def memoize(cb: EmitCodeBuilder, name: String): SBinaryValue

  def memoizeField(cb: EmitCodeBuilder, name: String): SBinaryValue
}


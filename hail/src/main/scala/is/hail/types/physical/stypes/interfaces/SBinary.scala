package is.hail.types.physical.stypes.interfaces

import is.hail.asm4s.Code
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.{SCode, SType, SValue}

trait SBinary extends SType

trait SBinaryValue extends SValue {
  def loadLength(): Code[Int]

  def loadBytes(): Code[Array[Byte]]

  def loadByte(i: Code[Int]): Code[Byte]
}

trait SBinaryCode extends SCode {
  def loadLength(): Code[Int]

  def loadBytes(): Code[Array[Byte]]

  def memoize(cb: EmitCodeBuilder, name: String): SBinaryValue

  def memoizeField(cb: EmitCodeBuilder, name: String): SBinaryValue
}


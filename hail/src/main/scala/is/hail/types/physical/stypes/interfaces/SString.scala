package is.hail.types.physical.stypes.interfaces

import is.hail.asm4s.Code
import is.hail.types.physical.stypes.{SCode, SType, SValue}

trait SString extends SType

trait SStringValue extends SValue

trait SStringCode extends SCode {
  def loadLength(): Code[Int]

  def loadString(): Code[String]

  def asBytes(): SBinaryCode
}

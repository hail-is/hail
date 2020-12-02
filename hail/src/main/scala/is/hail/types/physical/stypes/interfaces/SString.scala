package is.hail.types.physical.stypes.interfaces

import is.hail.asm4s.{Code, LineNumber}
import is.hail.types.physical.stypes.{SCode, SType, SValue}

trait SString extends SType

trait SStringValue extends SValue

trait SStringCode extends SCode {
  def loadLength()(implicit line: LineNumber): Code[Int]

  def loadString()(implicit line: LineNumber): Code[String]

  def asBytes(): SBinaryCode
}

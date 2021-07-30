package is.hail.types.physical.stypes.interfaces

import is.hail.annotations.Region
import is.hail.asm4s.{Code, Value}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.{SCode, SType, SValue}

trait SString extends SType {
  def constructFromString(cb: EmitCodeBuilder, r: Value[Region], s: Code[String]): SStringCode
}

trait SStringValue extends SValue {
  override def get: SStringCode
}

trait SStringCode extends SCode {
  def loadLength(): Code[Int]

  def loadString(): Code[String]

  def toBytes(): SBinaryCode
}

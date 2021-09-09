package is.hail.types.physical.stypes.interfaces

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.primitives.SInt32Code
import is.hail.types.physical.stypes.{SCode, SType, SValue}
import is.hail.types.{RPrimitive, TypeWithRequiredness}

trait SString extends SType {
  def constructFromString(cb: EmitCodeBuilder, r: Value[Region], s: Code[String]): SStringCode
  override def _typeWithRequiredness: TypeWithRequiredness = RPrimitive()
}

trait SStringValue extends SValue {
  override def get: SStringCode

  override def hash(cb: EmitCodeBuilder): SInt32Code = {
    new SInt32Code(get.loadString().invoke[Int]("hashCode"))
  }

  def loadLength(cb: EmitCodeBuilder): Value[Int]

  def loadString(cb: EmitCodeBuilder): Value[String]

  def toBytes(cb: EmitCodeBuilder): SBinaryValue
}

trait SStringCode extends SCode {
  def loadLength(): Code[Int]

  def loadString(): Code[String]

  def toBytes(): SBinaryCode
}

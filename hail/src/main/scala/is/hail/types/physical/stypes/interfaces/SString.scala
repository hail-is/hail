package is.hail.types.physical.stypes.interfaces

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.{RPrimitive, TypeWithRequiredness}
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.types.physical.stypes.primitives.SInt32Value

trait SString extends SType {
  def constructFromString(cb: EmitCodeBuilder, r: Value[Region], s: Code[String]): SStringValue
  override def _typeWithRequiredness: TypeWithRequiredness = RPrimitive()
}

trait SStringValue extends SValue {
  override def hash(cb: EmitCodeBuilder): SInt32Value =
    new SInt32Value(cb.memoize(loadString(cb).invoke[Int]("hashCode")))

  def loadLength(cb: EmitCodeBuilder): Value[Int]

  def loadString(cb: EmitCodeBuilder): Value[String]

  def toBytes(cb: EmitCodeBuilder): SBinaryValue
}

package is.hail.types.physical.stypes.primitives

import is.hail.asm4s._
import is.hail.types.{RPrimitive, TypeWithRequiredness}
import is.hail.types.physical.stypes.{SCode, SType, SValue}

trait SPrimitive extends SType {
  def ti: TypeInfo[_]

  override def _typeWithRequiredness: TypeWithRequiredness = RPrimitive()

  override def copiedType: SType = this

  def containsPointers: Boolean = false
}

abstract class SPrimitiveValue extends SValue {
  override def st: SPrimitive

  protected[primitives] def _primitiveValue: Value[_]
  final def primitiveValue[T]: Value[T] = coerce[T](_primitiveValue)
}

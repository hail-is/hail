package is.hail.types.physical.stypes.primitives

import is.hail.asm4s._
import is.hail.types.physical.stypes.{SCode, SType}

trait SPrimitive extends SType {
  def ti: TypeInfo[_]
}

trait SPrimitiveCode extends SCode {
  def st: SPrimitive

  protected[primitives] def _primitiveCode: Code[_]
  def primitiveCode[T]: Code[T] = coerce[T](_primitiveCode)
}

package is.hail.types.physical.stypes.primitives

import is.hail.asm4s._
import is.hail.asm4s.implicits._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.io.PrefixCoder
import is.hail.types.{RPrimitive, TypeWithRequiredness}
import is.hail.types.physical.stypes.{SType, SValue}

trait SPrimitive extends SType {
  def ti: TypeInfo[_]

  override def _typeWithRequiredness: TypeWithRequiredness = RPrimitive()

  override def copiedType: SType = this

  override def containsPointers: Boolean = false
}

abstract class SPrimitiveValue extends SValue {
  override def st: SPrimitive

  protected[primitives] def _primitiveValue: Value[_]
  final def primitiveValue[T]: Value[T] = coerce[T](_primitiveValue)

  override def prefixCode(cb: EmitCodeBuilder, pc: Value[PrefixCoder]) = st match {
    case SBoolean => pc.encodeBool(cb, primitiveValue[Boolean])
    case SInt32 => pc.encodeInt(cb, primitiveValue[Int])
    case SInt64 => pc.encodeLong(cb, primitiveValue[Long])
    case SFloat32 => pc.encodeFloat(cb, primitiveValue[Float])
    case SFloat64 => pc.encodeDouble(cb, primitiveValue[Double])
  }
}

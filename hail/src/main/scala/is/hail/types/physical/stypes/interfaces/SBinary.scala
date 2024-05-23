package is.hail.types.physical.stypes.interfaces

import is.hail.asm4s._
import is.hail.asm4s.Code.invokeStatic1
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.{RPrimitive, TypeWithRequiredness}
import is.hail.types.physical.PCanonicalBinary
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.types.physical.stypes.primitives.{SInt32Value, SInt64Value}

trait SBinary extends SType {
  override def _typeWithRequiredness: TypeWithRequiredness = RPrimitive()
}

trait SBinaryValue extends SValue {
  def loadLength(cb: EmitCodeBuilder): Value[Int]

  def loadBytes(cb: EmitCodeBuilder): Value[Array[Byte]]

  def loadByte(cb: EmitCodeBuilder, i: Code[Int]): Value[Byte]

  override def hash(cb: EmitCodeBuilder): SInt32Value =
    new SInt32Value(cb.memoize(invokeStatic1[java.util.Arrays, Array[Byte], Int](
      "hashCode",
      loadBytes(cb),
    )))

  override def sizeToStoreInBytes(cb: EmitCodeBuilder): SInt64Value = {
    val binaryStorageType = this.st.storageType().asInstanceOf[PCanonicalBinary]
    val contentsByteSize = binaryStorageType.contentByteSize(this.loadLength(cb))
    new SInt64Value(cb.memoize(contentsByteSize))
  }
}

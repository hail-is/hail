package is.hail.types.arrowextension

import org.apache.arrow.memory.BufferAllocator
import org.apache.arrow.vector.FieldVector
import org.apache.arrow.vector.types.pojo.{ArrowType, ExtensionTypeRegistry, FieldType}

object NDArray {
  val INSTANCE = new NDArray()

  ExtensionTypeRegistry.register(INSTANCE)
}

class NDArray private () extends ArrowType.ExtensionType {
  override def extensionName(): String = "hail.ndarray"

  override def extensionEquals(other: ArrowType.ExtensionType): Boolean =
    other.isInstanceOf[NDArray]

  override def storageType(): ArrowType = new ArrowType.Struct()

  override def serialize(): String = ""

  override def deserialize(storageType: ArrowType, serializedData: String): ArrowType =
    if (!storageType.equals(this.storageType()))
      throw new UnsupportedOperationException("bad storage type: " + storageType)
    else
      NDArray.INSTANCE

  override def getNewVector(name: String, fieldType: FieldType, allocator: BufferAllocator)
    : FieldVector = ???
}

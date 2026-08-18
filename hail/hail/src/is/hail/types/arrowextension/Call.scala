package is.hail.types.arrowextension

import org.apache.arrow.memory.BufferAllocator
import org.apache.arrow.vector.FieldVector
import org.apache.arrow.vector.types.pojo.{ArrowType, ExtensionTypeRegistry, FieldType}

object Call {
  val INSTANCE = new Call()

  ExtensionTypeRegistry.register(INSTANCE)
}

class Call private () extends ArrowType.ExtensionType {
  override def extensionName(): String = "hail.call"

  override def extensionEquals(other: ArrowType.ExtensionType): Boolean = other.isInstanceOf[Call]

  override def storageType(): ArrowType = new ArrowType.Int(32, false)

  override def serialize(): String = ""

  override def deserialize(storageType: ArrowType, serializedData: String): ArrowType =
    if (!storageType.equals(this.storageType()))
      throw new UnsupportedOperationException("bad storage type: " + storageType)
    else
      Call.INSTANCE

  override def getNewVector(name: String, fieldType: FieldType, allocator: BufferAllocator)
    : FieldVector = ???
}

package is.hail.types.arrowextension

import org.apache.arrow.memory.BufferAllocator
import org.apache.arrow.vector.FieldVector
import org.apache.arrow.vector.types.pojo.{ArrowType, ExtensionTypeRegistry, FieldType}

object Locus {
  val INSTANCE = new Locus()

  ExtensionTypeRegistry.register(INSTANCE)
}

class Locus private () extends ArrowType.ExtensionType {
  override def extensionName(): String = "hail.locus"

  override def extensionEquals(other: ArrowType.ExtensionType): Boolean = other.isInstanceOf[Locus]

  override def storageType(): ArrowType = new ArrowType.Struct()

  override def serialize(): String = ""

  override def deserialize(storageType: ArrowType, serializedData: String): ArrowType =
    if (!storageType.equals(this.storageType()))
      throw new UnsupportedOperationException("bad storage type: " + storageType)
    else
      Locus.INSTANCE

  override def getNewVector(name: String, fieldType: FieldType, allocator: BufferAllocator)
    : FieldVector = ???
}

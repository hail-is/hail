package is.hail.types.arrow

import org.apache.arrow.memory.BufferAllocator
import org.apache.arrow.vector.FieldVector
import org.apache.arrow.vector.extension.InvalidExtensionMetadataException
import org.apache.arrow.vector.types.pojo.{ArrowType, ExtensionTypeRegistry, FieldType}
import org.json4s._
import org.json4s.jackson.JsonMethods

/** Arrow extension types used by the columnar format.
  *
  * Extension metadata is a JSON object. Readers ignore keys they do not recognise, so keys can be
  * added without breaking existing readers.
  *
  * The columnar format is written without Arrow vectors, so none of these types can create one.
  */
object ExtensionTypes {
  private lazy val registered: Unit =
    Seq[ArrowType.ExtensionType](
      CallExtensionType.Current,
      LocusExtensionType.Prototype,
      SetExtensionType,
      IntervalExtensionType,
      TupleExtensionType,
      VariableShapeTensorExtensionType,
    ).foreach(ExtensionTypeRegistry.register)

  /** Registers every extension type with Arrow's global registry, so that schemas read by Arrow
    * contain extension types rather than their storage types. Idempotent.
    */
  def register(): Unit = registered
}

sealed abstract class HailExtensionType extends ArrowType.ExtensionType {
  protected def checkStorageType(storageType: ArrowType): Unit =
    if (storageType != this.storageType())
      throw new InvalidExtensionMetadataException(
        s"${extensionName()}: expected storage type ${this.storageType()}, found $storageType"
      )

  protected def parseMetadata(serializedData: String): JObject =
    try
      JsonMethods.parse(serializedData) match {
        case o: JObject => o
        case _ => throw new InvalidExtensionMetadataException(
            s"${extensionName()}: metadata is not a JSON object: $serializedData"
          )
      }
    catch {
      case e: com.fasterxml.jackson.core.JsonProcessingException =>
        throw new InvalidExtensionMetadataException(
          s"${extensionName()}: metadata is not valid JSON: $serializedData",
          e,
        )
    }

  override def getNewVector(name: String, fieldType: FieldType, allocator: BufferAllocator)
    : FieldVector =
    throw new UnsupportedOperationException(
      s"Arrow vectors are not implemented for extension type ${extensionName()}"
    )
}

/** An extension type with no parameters. Everything needed to reconstruct the Hail type is in the
  * storage type and the field's children.
  */
sealed abstract class MarkerExtensionType(name: String, storage: ArrowType)
    extends HailExtensionType {
  override def extensionName(): String = name

  override def storageType(): ArrowType = storage

  override def extensionEquals(other: ArrowType.ExtensionType): Boolean = other eq this

  override def serialize(): String = "{}"

  override def deserialize(storageType: ArrowType, serializedData: String): ArrowType = {
    checkStorageType(storageType)
    parseMetadata(serializedData): Unit
    this
  }
}

/** `list<T>` whose elements are sorted and distinct. */
object SetExtensionType extends MarkerExtensionType("hail.set", ArrowType.List.INSTANCE)

/** `struct<start: T, end: T, includes_start: bool, includes_end: bool>`. */
object IntervalExtensionType extends MarkerExtensionType("hail.interval", ArrowType.Struct.INSTANCE)

/** A struct whose field names are the tuple indices `"0"`, `"1"`, .... */
object TupleExtensionType extends MarkerExtensionType("hail.tuple", ArrowType.Struct.INSTANCE)

/** Arrow's canonical variable shape tensor: `struct<data: list<T>, shape:
  * fixed_size_list<int32>[ndim]>`, row-major. arrow-java does not provide it.
  */
object VariableShapeTensorExtensionType
    extends MarkerExtensionType("arrow.variable_shape_tensor", ArrowType.Struct.INSTANCE)

object CallExtensionType {
  val Name: String = "hail.call"

  /** Hail's packed call representation, as produced by `is.hail.variant.Call`. */
  val Current: CallExtensionType = new CallExtensionType(1)
}

/** A Hail call packed into a signed `int32`. `encoding` identifies the bit layout. */
final class CallExtensionType(val encoding: Int) extends HailExtensionType {
  override def extensionName(): String = CallExtensionType.Name

  override def storageType(): ArrowType = new ArrowType.Int(32, true)

  override def extensionEquals(other: ArrowType.ExtensionType): Boolean =
    other match {
      case o: CallExtensionType => o.encoding == encoding
      case _ => false
    }

  override def serialize(): String =
    JsonMethods.compact(JObject("encoding" -> JInt(encoding)))

  override def deserialize(storageType: ArrowType, serializedData: String): ArrowType = {
    checkStorageType(storageType)
    parseMetadata(serializedData) \ "encoding" match {
      case JInt(e) if e.isValidInt => new CallExtensionType(e.toInt)
      case _ => throw new InvalidExtensionMetadataException(
          s"$extensionName: missing or invalid 'encoding': $serializedData"
        )
    }
  }

  override def toString: String = s"CallExtensionType(encoding=$encoding)"
}

object LocusExtensionType {
  val Name: String = "hail.locus"

  private[arrow] val Prototype: LocusExtensionType = new LocusExtensionType("", IndexedSeq.empty)
}

/** A locus packed into a signed `int64` as `(contig_index << 32) | position`, where `contig_index`
  * indexes `contigs`, the reference genome's contigs in order.
  */
final class LocusExtensionType(val referenceGenome: String, val contigs: IndexedSeq[String])
    extends HailExtensionType {
  override def extensionName(): String = LocusExtensionType.Name

  override def storageType(): ArrowType = new ArrowType.Int(64, true)

  override def extensionEquals(other: ArrowType.ExtensionType): Boolean =
    other match {
      case o: LocusExtensionType => o.referenceGenome == referenceGenome && o.contigs == contigs
      case _ => false
    }

  override def serialize(): String =
    JsonMethods.compact(JObject(
      "reference_genome" -> JString(referenceGenome),
      "contigs" -> JArray(contigs.map(JString(_)).toList),
    ))

  override def deserialize(storageType: ArrowType, serializedData: String): ArrowType = {
    checkStorageType(storageType)
    val metadata = parseMetadata(serializedData)
    val rg = metadata \ "reference_genome" match {
      case JString(rg) => rg
      case _ => throw new InvalidExtensionMetadataException(
          s"$extensionName: missing or invalid 'reference_genome'"
        )
    }
    val contigs = metadata \ "contigs" match {
      case JArray(cs) => cs.toIndexedSeq.map {
          case JString(c) => c
          case _ => throw new InvalidExtensionMetadataException(
              s"$extensionName: 'contigs' must be an array of strings"
            )
        }
      case _ => throw new InvalidExtensionMetadataException(
          s"$extensionName: missing or invalid 'contigs'"
        )
    }
    new LocusExtensionType(rg, contigs)
  }

  // Deliberately omits the contig list, which can run to thousands of entries.
  override def toString: String = s"LocusExtensionType($referenceGenome)"
}

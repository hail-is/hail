package is.hail.compatibility

import is.hail.collection.FastSeq
import is.hail.expr.JSONAnnotationImpex
import is.hail.io._
import is.hail.rvd.{AbstractRVDSpec, IndexSpec2, IndexedRVDSpec2, RVDPartitioner}
import is.hail.types.encoded._
import is.hail.types.virtual._
import is.hail.utils.Interval

import org.json4s.JValue

case class IndexSpec private (
  relPath: String,
  keyType: LegacyEncodedType,
  annotationType: LegacyEncodedType,
  offsetField: Option[String],
) {
  val baseSpec = LEB128BufferSpec(
    BlockingBufferSpec(32 * 1024, LZ4BlockBufferSpec(32 * 1024, new StreamBlockBufferSpec))
  )

  val keyVType = keyType.vType
  val keyEType = keyType.eType
  val annotationVType = annotationType.vType
  val annotationEType = annotationType.eType

  val leafEType = EBaseStruct(FastSeq(
    EField("first_idx", EInt64Required, 0),
    EField(
      "keys",
      EArrayLegacyFullWidthIntegerLength(
        EBaseStruct(
          FastSeq(
            EField("key", keyEType, 0),
            EField("offset", EInt64Required, 1),
            EField("annotation", annotationEType, 2),
          ),
          required = true,
        ),
        required = true,
      ),
      1,
    ),
  ))

  val leafVType = TStruct(FastSeq(
    Field("first_idx", TInt64, 0),
    Field(
      "keys",
      TArray(TStruct(FastSeq(
        Field("key", keyVType, 0),
        Field("offset", TInt64, 1),
        Field("annotation", annotationVType, 2),
      ))),
      1,
    ),
  ))

  val internalNodeEType = EBaseStruct(FastSeq(
    EField(
      "children",
      EArrayLegacyFullWidthIntegerLength(
        EBaseStruct(
          FastSeq(
            EField("index_file_offset", EInt64Required, 0),
            EField("first_idx", EInt64Required, 1),
            EField("first_key", keyEType, 2),
            EField("first_record_offset", EInt64Required, 3),
            EField("first_annotation", annotationEType, 4),
          ),
          required = true,
        ),
        required = true,
      ),
      0,
    )
  ))

  val internalNodeVType = TStruct(FastSeq(
    Field(
      "children",
      TArray(TStruct(FastSeq(
        Field("index_file_offset", TInt64, 0),
        Field("first_idx", TInt64, 1),
        Field("first_key", keyVType, 2),
        Field("first_record_offset", TInt64, 3),
        Field("first_annotation", annotationVType, 4),
      ))),
      0,
    )
  ))

  val leafCodec: AbstractTypedCodecSpec = TypedCodecSpec(leafEType, leafVType, baseSpec)

  val internalNodeCodec: AbstractTypedCodecSpec =
    TypedCodecSpec(internalNodeEType, internalNodeVType, baseSpec)

  def toIndexSpec2: IndexSpec2 = IndexSpec2(
    relPath, leafCodec, internalNodeCodec, keyVType, annotationVType, offsetField,
  )
}

case class PackCodecSpec private (child: BufferSpec)

/** An RVD type written in the legacy encoded type syntax, and the string it was parsed from. */
case class LegacyRVDType(repr: String, rowType: TStruct, rowEType: EType, key: IndexedSeq[String]) {
  def keyType: TStruct = rowType.select(key)._1
}

trait ShimRVDSpec extends AbstractRVDSpec {

  val shim: AbstractRVDSpec

  final override def key: IndexedSeq[String] = shim.key

  override lazy val partitioner: RVDPartitioner = shim.partitioner

  override def typedCodecSpec: AbstractTypedCodecSpec = shim.typedCodecSpec

  override def partFiles: IndexedSeq[String] = shim.partFiles

  override lazy val indexed: Boolean = shim.indexed

  lazy val attrs: Map[String, String] = shim.attrs
}

case class IndexedRVDSpec private (
  rvdType: LegacyRVDType,
  codecSpec: PackCodecSpec,
  indexSpec: IndexSpec,
  override val partFiles: IndexedSeq[String],
  jRangeBounds: JValue,
) extends ShimRVDSpec {
  lazy val shim = IndexedRVDSpec2(
    rvdType.key,
    TypedCodecSpec(rvdType.rowEType.setRequired(true), rvdType.rowType, codecSpec.child),
    indexSpec.toIndexSpec2,
    partFiles,
    jRangeBounds,
    Map.empty[String, String],
  )
}

case class UnpartitionedRVDSpec private (
  rowType: LegacyEncodedType,
  codecSpec: PackCodecSpec,
  partFiles: IndexedSeq[String],
) extends AbstractRVDSpec {
  private val rowVType = rowType.vType.asInstanceOf[TStruct]
  private val rowEType = rowType.eType

  override lazy val partitioner: RVDPartitioner =
    RVDPartitioner.unkeyed(partFiles.length)

  override def key: IndexedSeq[String] = FastSeq()

  override def typedCodecSpec: AbstractTypedCodecSpec =
    TypedCodecSpec(rowEType.setRequired(true), rowVType, codecSpec.child)

  val attrs: Map[String, String] = Map.empty
}

case class OrderedRVDSpec private (
  rvdType: LegacyRVDType,
  codecSpec: PackCodecSpec,
  partFiles: IndexedSeq[String],
  jRangeBounds: JValue,
) extends AbstractRVDSpec {
  override def key: IndexedSeq[String] = rvdType.key

  override def partitioner: RVDPartitioner = {
    val rangeBoundsType = TArray(TInterval(rvdType.keyType))
    new RVDPartitioner(
      rvdType.keyType,
      JSONAnnotationImpex.importAnnotation(
        jRangeBounds,
        rangeBoundsType,
        padNulls = false,
      ).asInstanceOf[IndexedSeq[Interval]],
    )
  }

  override def typedCodecSpec: AbstractTypedCodecSpec =
    TypedCodecSpec(rvdType.rowEType.setRequired(true), rvdType.rowType, codecSpec.child)

  val attrs: Map[String, String] = Map.empty
}

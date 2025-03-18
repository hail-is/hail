package is.hail.compatibility

import is.hail.backend.ExecuteContext
import is.hail.compatibility.LegacyEncodedTypeParser.{parseLegacyRVDType, parseTypeAndEType}
import is.hail.expr.JSONAnnotationImpex
import is.hail.io._
import is.hail.rvd.{AbstractRVDSpec, IndexSpec2, IndexedRVDSpec2, RVDPartitioner}
import is.hail.types.encoded._
import is.hail.types.virtual._
import is.hail.utils.{FastSeq, Interval, Json4sFormat, Json4sReader, Json4sWriter}
import org.json4s.Extraction.{decompose, extract}
import org.json4s.{Formats, JObject, JString, JValue, ShortTypeHints, TypeHints}
import is.hail.utils.json4s._

object IndexSpec extends Json4sFormat[IndexSpec, JObject] {
  override lazy val hints: TypeHints =
    ShortTypeHints(classOf[IndexSpec] :: Nil, typeHintFieldName = "name")

  override lazy val writer: Json4sWriter[IndexSpec, JObject] =
    (s: IndexSpec) => {
      implicit formats: Formats =>
        JObject(
          "name" -> JString(classOf[IndexSpec].getSimpleName),
          "relPath" -> JString(s.relPath),
          "keyType" -> JString(s.keyVType._toPretty),
          "annotationType" -> JString(s.annotationVType._toPretty),
          "offsetField" -> decompose(s.offsetField),
        )
    }

  override lazy val reader: Json4sReader[IndexSpec, JObject] =
    (ctx: ExecuteContext, v: JObject) => {
      implicit formats: Formats =>
        val (kVType, kEType) = parseTypeAndEType(ctx, (v \ "keyType").extract[String])
        val (aVType, aEType) = parseTypeAndEType(ctx, (v \ "annotationType").extract[String])
        IndexSpec(
          relPath = extract(v \ "relPath"),
          keyVType = kVType,
          keyEType = kEType,
          annotationVType = aVType,
          annotationEType = aEType,
          offsetField = extract(v \ "offsetField"),
        )
    }
}

case class IndexSpec private (
  relPath: String,
  keyVType: Type,
  keyEType: EType,
  annotationVType: Type,
  annotationEType: EType,
  offsetField: Option[String],
) {
  val baseSpec = LEB128BufferSpec(
    BlockingBufferSpec(32 * 1024, LZ4BlockBufferSpec(32 * 1024, new StreamBlockBufferSpec))
  )

  val leafEType = EBaseStruct(FastSeq(
    EField("first_idx", EInt64Required, 0),
    EField(
      "keys",
      EArray(
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
      EArray(
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

case class LegacyRVDType(rowType: TStruct, rowEType: EType, key: IndexedSeq[String]) {
  def keyType: TStruct = rowType.select(key)._1

  private[this] def pretty(tycon: String): String =
    f"$tycon{key:[${key.mkString("[", ",", "]")}],row:${rowType._toPretty}}"

  def _toPretty: String = pretty("RVDType")
  def _toOrderedPretty: String = pretty("OrderedRVDType")
}

trait ShimRVDSpec extends AbstractRVDSpec {

  val shim: AbstractRVDSpec

  final def key: IndexedSeq[String] = shim.key

  override def partitioner: RVDPartitioner = shim.partitioner

  override def typedCodecSpec: AbstractTypedCodecSpec = shim.typedCodecSpec

  override def partFiles: Array[String] = shim.partFiles

  override lazy val indexed: Boolean = shim.indexed

  lazy val attrs: Map[String, String] = shim.attrs
}

object IndexedRVDSpec extends Json4sFormat[IndexedRVDSpec, JObject] {
  override lazy val hints: TypeHints =
    ShortTypeHints(classOf[IndexedRVDSpec] :: Nil, typeHintFieldName = "name")

  override lazy val reader: Json4sReader[IndexedRVDSpec, JObject] =
    (ctx: ExecuteContext, v: JObject) => {
      implicit formats: Formats =>
        IndexedRVDSpec(
          rvdType = parseLegacyRVDType(ctx, (v \ "rvdType").extract[String]),
          codecSpec = extract(v \ "codes"),
          indexSpec = IndexSpec.reader(ctx, (v \ "indexSpec").extract[JObject]),
          partFiles = (v \ "partFiles").extract[Array[String]],
          jRangeBounds = v \ "jRangeBounds",
        )
    }

  override lazy val writer: Json4sWriter[IndexedRVDSpec, JObject] =
    (s: IndexedRVDSpec) => {
      implicit formats: Formats =>
        JObject(
          "name" -> JString(classOf[IndexedRVDSpec].getSimpleName),
          "rvdType" -> JString(s.rvdType._toPretty),
          "codecSpec" -> decompose(s.codecSpec),
          "indexSpec" -> IndexSpec.writer(s.indexSpec),
          "partFiles" -> decompose(s.partFiles),
          "jRangeBounds" -> s.jRangeBounds,
        )
    }
}

case class IndexedRVDSpec private (
  rvdType: LegacyRVDType,
  codecSpec: PackCodecSpec,
  indexSpec: IndexSpec,
  override val partFiles: Array[String],
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

object UnpartitionedRVDSpec extends Json4sFormat[UnpartitionedRVDSpec, JObject] {
  override lazy val hints: TypeHints =
    ShortTypeHints(classOf[UnpartitionedRVDSpec] :: Nil, typeHintFieldName = "name")

  override lazy val reader: Json4sReader[UnpartitionedRVDSpec, JObject] =
    (ctx: ExecuteContext, v: JObject) => {
      implicit formats: Formats =>
        val (rowType, rowEType) = parseTypeAndEType(ctx, (v \ "rowType").extract[String])
        UnpartitionedRVDSpec(
          rowVType = rowType,
          rowEType = rowEType,
          codecSpec = (v \ "codecSpec").extract[PackCodecSpec],
          partFiles = (v \ "partFiles").extract[Array[String]],
        )
    }

  override lazy val writer: Json4sWriter[UnpartitionedRVDSpec, JObject] =
    (a: UnpartitionedRVDSpec) => {
      implicit formats: Formats =>
        JObject(
          "name" -> JString(classOf[UnpartitionedRVDSpec].getSimpleName),
          "rowType" -> JString(a.rowVType._toPretty),
          "codecSpec" -> decompose(a.codecSpec),
          "partFiles" -> decompose(a.partFiles),
        )
    }
}

case class UnpartitionedRVDSpec private (
  rowVType: Type,
  rowEType: EType,
  codecSpec: PackCodecSpec,
  partFiles: Array[String],
) extends AbstractRVDSpec {

  def partitioner: RVDPartitioner =
    RVDPartitioner.unkeyed(partFiles.length)

  def key: IndexedSeq[String] = FastSeq()

  def typedCodecSpec: AbstractTypedCodecSpec =
    TypedCodecSpec(rowEType.setRequired(true), rowVType, codecSpec.child)

  val attrs: Map[String, String] = Map.empty
}

object OrderedRVDSpec extends Json4sFormat[OrderedRVDSpec, JObject] {
  override lazy val hints: TypeHints =
    ShortTypeHints(classOf[OrderedRVDSpec] :: Nil, typeHintFieldName = "name")

  override lazy val reader: Json4sReader[OrderedRVDSpec, JObject] =
    (ctx: ExecuteContext, v: JObject) => {
      implicit formats: Formats =>
        OrderedRVDSpec(
          rvdType = parseLegacyRVDType(ctx, (v \ "rvdType").extract[String]),
          codecSpec = extract(v \ "codecSpec"),
          partFiles = extract(v \ "partFiles"),
          jRangeBounds = v \ "jRangeBounds",
        )
    }

  override lazy val writer: Json4sWriter[OrderedRVDSpec, JObject] =
    (a: OrderedRVDSpec) => {
      implicit formats: Formats =>
        JObject(
          "name" -> JString(classOf[OrderedRVDSpec].getSimpleName),
          "rvdType" -> JString(a.rvdType._toOrderedPretty),
          "codecSpec" -> decompose(a.codecSpec),
          "partFiles" -> decompose(a.partFiles),
          "jRangeBounds" -> a.jRangeBounds,
        )
    }
}

case class OrderedRVDSpec private (
  rvdType: LegacyRVDType,
  codecSpec: PackCodecSpec,
  partFiles: Array[String],
  jRangeBounds: JValue,
) extends AbstractRVDSpec {
  def key: IndexedSeq[String] = rvdType.key

  def partitioner: RVDPartitioner = {
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

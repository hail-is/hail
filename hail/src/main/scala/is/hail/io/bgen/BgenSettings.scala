package is.hail.io.bgen

import is.hail.backend.BroadcastValue
import is.hail.expr.ir.PruneDeadFields
import is.hail.io._
import is.hail.types.encoded._
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.types.{MatrixType, TableType}
import is.hail.utils._
import is.hail.variant.ReferenceGenome


object BgenSettings {
  val UNCOMPRESSED: Int = 0
  val ZLIB_COMPRESSION: Int = 1
  val ZSTD_COMPRESSION: Int = 2

  def indexKeyType(rg: Option[String]): TStruct = TStruct(
    "locus" -> rg.map(TLocus(_)).getOrElse(TLocus.representation),
    "alleles" -> TArray(TString))

  val indexAnnotationType: Type = TStruct.empty

  private def specFromVersion(indexVersion: SemanticVersion): BufferSpec =
    if (indexVersion >= SemanticVersion(1, 2, 0)) {
      BufferSpec.zstdCompressionLEB
    } else {
      BufferSpec.lz4HCCompressionLEB
    }


  def indexCodecSpecs(indexVersion: SemanticVersion, rg: Option[String]): (AbstractTypedCodecSpec, AbstractTypedCodecSpec) = {
    val bufferSpec = specFromVersion(indexVersion)

    val keyVType = indexKeyType(rg)
    val keyEType = EBaseStruct(FastIndexedSeq(
      EField("locus", EBaseStruct(FastIndexedSeq(
        EField("contig", EBinaryRequired, 0),
        EField("position", EInt32Required, 1))), 0),
      EField("alleles", EArray(EBinaryOptional, required = false), 1)),
      required = false)

    val annotationVType = TStruct.empty
    val annotationEType = EBaseStruct(FastIndexedSeq(), required = true)

    val leafEType = EBaseStruct(FastIndexedSeq(
      EField("first_idx", EInt64Required, 0),
      EField("keys", EArray(EBaseStruct(FastIndexedSeq(
        EField("key", keyEType, 0),
        EField("offset", EInt64Required, 1),
        EField("annotation", annotationEType, 2)
      ), required = true), required = true), 1)
    ))
    val leafVType = TStruct(FastIndexedSeq(
      Field("first_idx", TInt64, 0),
      Field("keys", TArray(TStruct(FastIndexedSeq(
        Field("key", keyVType, 0),
        Field("offset", TInt64, 1),
        Field("annotation", annotationVType, 2)
      ))), 1)))

    val internalNodeEType = EBaseStruct(FastIndexedSeq(
      EField("children", EArray(EBaseStruct(FastIndexedSeq(
        EField("index_file_offset", EInt64Required, 0),
        EField("first_idx", EInt64Required, 1),
        EField("first_key", keyEType, 2),
        EField("first_record_offset", EInt64Required, 3),
        EField("first_annotation", annotationEType, 4)
      ), required = true), required = true), 0)
    ))

    val internalNodeVType = TStruct(FastIndexedSeq(
      Field("children", TArray(TStruct(FastIndexedSeq(
        Field("index_file_offset", TInt64, 0),
        Field("first_idx", TInt64, 1),
        Field("first_key", keyVType, 2),
        Field("first_record_offset", TInt64, 3),
        Field("first_annotation", annotationVType, 4)
      ))), 0)
    ))

    (TypedCodecSpec(leafEType, leafVType, bufferSpec), (TypedCodecSpec(internalNodeEType, internalNodeVType, bufferSpec)))
  }
}

case class BgenSettings(
  nSamples: Int,
  requestedType: TableType,
  rg: Option[String],
  indexAnnotationType: Type
) {
  require(PruneDeadFields.isSupertype(requestedType, MatrixBGENReader.fullMatrixType(rg).canonicalTableType))

  val entryType: Option[TStruct] = requestedType.rowType
    .fieldOption(MatrixType.entriesIdentifier)
    .map(f => f.typ.asInstanceOf[TArray].elementType.asInstanceOf[TStruct])

  val rowPType: PCanonicalStruct = PCanonicalStruct(required = true,
    Array(
      "locus" -> PCanonicalLocus.schemaFromRG(rg, required = false),
      "alleles" -> PCanonicalArray(PCanonicalString(false), false),
      "rsid" -> PCanonicalString(),
      "varid" -> PCanonicalString(),
      "offset" -> PInt64(),
      "file_idx" -> PInt32(),
      MatrixType.entriesIdentifier -> PCanonicalArray(PCanonicalStruct(
        Array(
          "GT" -> PCanonicalCall(),
          "GP" -> PCanonicalArray(PFloat64Required, required = true),
          "dosage" -> PFloat64Required
        ).filter { case (name, _) => entryType.exists(t => t.hasField(name))
        }: _*)))
      .filter { case (name, _) => requestedType.rowType.hasField(name) }: _*)

  assert(rowPType.virtualType == requestedType.rowType, s"${ rowPType.virtualType.parsableString() } vs ${ requestedType.rowType.parsableString() }")

  val indexKeyType: PStruct = rowPType.selectFields(Array("locus", "alleles")).setRequired(false).asInstanceOf[PStruct]

  def hasField(name: String): Boolean = requestedType.rowType.hasField(name)

  def hasEntryField(name: String): Boolean = entryType.exists(t => t.hasField(name))
}

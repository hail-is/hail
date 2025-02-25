package is.hail.io.bgen

import is.hail.expr.ir.PruneDeadFields
import is.hail.io._
import is.hail.rvd.AbstractIndexSpec
import is.hail.types.encoded._
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.utils._

object BgenSettings {
  val UNCOMPRESSED: Int = 0
  val ZLIB_COMPRESSION: Int = 1
  val ZSTD_COMPRESSION: Int = 2

  def indexKeyType(rg: Option[String]): TStruct = TStruct(
    "locus" -> rg.map(TLocus(_)).getOrElse(TLocus.representation),
    "alleles" -> TArray(TString),
  )

  val indexAnnotationType: Type = TStruct.empty

  private def specFromVersion(indexVersion: SemanticVersion): BufferSpec =
    if (indexVersion >= SemanticVersion(1, 2, 0)) {
      BufferSpec.zstdCompressionLEB
    } else {
      BufferSpec.lz4HCCompressionLEB
    }

  def getIndexSpec(indexVersion: SemanticVersion, rg: Option[String]): AbstractIndexSpec = {
    val bufferSpec = specFromVersion(indexVersion)

    val keyVType = indexKeyType(rg)
    val keyEType = EBaseStruct(
      FastSeq(
        EField(
          "locus",
          EBaseStruct(FastSeq(
            EField("contig", EBinaryRequired, 0),
            EField("position", EInt32Required, 1),
          )),
          0,
        ),
        EField("alleles", EArray(EBinaryOptional, required = false), 1),
      ),
      required = false,
    )

    val annotationVType = TStruct.empty
    val annotationEType = EBaseStruct(FastSeq(), required = true)

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

    new AbstractIndexSpec {
      def relPath = fatal("relPath called for bgen index spec")
      val leafCodec = TypedCodecSpec(leafEType, leafVType, bufferSpec)
      val internalNodeCodec = TypedCodecSpec(internalNodeEType, internalNodeVType, bufferSpec)
      val keyType = keyVType
      val annotationType = annotationVType
    }
  }
}

case class BgenSettings(
  nSamples: Int,
  requestedType: TableType,
  rg: Option[String],
  indexAnnotationType: Type,
) {
  require(PruneDeadFields.isSupertype(
    requestedType,
    MatrixBGENReader.fullMatrixType(rg).canonicalTableType,
  ))

  val entryType: Option[TStruct] = requestedType.rowType
    .selfField(MatrixType.entriesIdentifier)
    .map(f => f.typ.asInstanceOf[TArray].elementType.asInstanceOf[TStruct])

  val rowPType: PCanonicalStruct = PCanonicalStruct(
    required = true,
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
          "dosage" -> PFloat64Required,
        ).filter { case (name, _) => entryType.exists(t => t.hasField(name)) }: _*
      )),
    )
      .filter { case (name, _) => requestedType.rowType.hasField(name) }: _*
  )

  assert(
    rowPType.virtualType == requestedType.rowType,
    s"${rowPType.virtualType.parsableString()} vs ${requestedType.rowType.parsableString()}",
  )

  val indexKeyType: PStruct =
    rowPType.selectFields(Array("locus", "alleles")).setRequired(false).asInstanceOf[PStruct]

  def hasField(name: String): Boolean = requestedType.rowType.hasField(name)

  def hasEntryField(name: String): Boolean = entryType.exists(t => t.hasField(name))
}

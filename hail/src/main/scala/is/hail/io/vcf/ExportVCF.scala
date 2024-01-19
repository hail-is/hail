package is.hail.io.vcf

import is.hail.io.{VCFAttributes, VCFFieldAttributes, VCFMetadata}
import is.hail.io.compress.{BGzipLineReader, BGzipOutputStream}
import is.hail.io.fs.FS
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.variant.ReferenceGenome

import htsjdk.samtools.util.FileExtensions
import htsjdk.tribble.SimpleFeature
import htsjdk.tribble.index.tabix.{TabixFormat, TabixIndexCreator}
import is.hail

object ExportVCF {
  def infoNumber(t: Type): String = t match {
    case TBoolean => "0"
    case TArray(_) => "."
    case TSet(_) => "."
    case _ => "1"
  }

  def fmtFloat(fmt: String, value: Float): String = fmt.format(value)
  def fmtDouble(fmt: String, value: Double): String = fmt.format(value)

  def infoType(t: Type): Option[String] = t match {
    case TInt32 | TInt64 => Some("Integer")
    case TFloat64 | TFloat32 => Some("Float")
    case TString => Some("String")
    case TBoolean => Some("Flag")
    case _ => None
  }

  def infoType(f: Field): String = {
    val tOption = f.typ match {
      case TArray(TBoolean) | TSet(TBoolean) => None
      case TArray(elt) => infoType(elt)
      case TSet(elt) => infoType(elt)
      case t => infoType(t)
    }
    tOption match {
      case Some(s) => s
      case _ => fatal(s"INFO field '${f.name}': VCF does not support type '${f.typ}'.")
    }
  }

  def formatType(t: Type): Option[String] = t match {
    case TInt32 | TInt64 => Some("Integer")
    case TFloat64 | TFloat32 => Some("Float")
    case TString => Some("String")
    case TCall => Some("String")
    case _ => None
  }

  def formatType(fieldName: String, t: Type): String = {
    val tOption = t match {
      case TArray(elt) => formatType(elt)
      case TSet(elt) => formatType(elt)
      case _ => formatType(t)
    }

    tOption match {
      case Some(s) => s
      case _ => fatal(s"FORMAT field '$fieldName': VCF does not support type '$t'.")
    }
  }

  def validInfoType(typ: Type): Boolean = {
    typ match {
      case TString => true
      case TFloat64 => true
      case TFloat32 => true
      case TInt32 => true
      case TInt64 => true
      case TBoolean => true
      case _ => false
    }
  }

  def checkInfoSignature(ti: TStruct): Unit = {
    val invalid = ti.fields.flatMap { fd =>
      val valid = fd.typ match {
        case it: TContainer if it.elementType != TBoolean => validInfoType(it.elementType)
        case t => validInfoType(t)
      }
      if (valid) {
        None
      } else {
        Some(s"\t'${fd.name}': '${fd.typ}'.")
      }
    }
    if (!invalid.isEmpty) {
      fatal(
        "VCF does not support the type(s) for the following INFO field(s):\n" + invalid.mkString(
          "\n"
        )
      )
    }
  }

  def validFormatType(typ: Type): Boolean = {
    typ match {
      case TString => true
      case TFloat64 => true
      case TFloat32 => true
      case TInt32 => true
      case TInt64 => true
      case TCall => true
      case _ => false
    }
  }

  def checkFormatSignature(tg: TStruct): Unit = {
    val invalid = tg.fields.flatMap { fd =>
      val valid = fd.typ match {
        case it: TContainer => validFormatType(it.elementType)
        case t => validFormatType(t)
      }
      if (valid) {
        None
      } else {
        Some(s"\t'${fd.name}': '${fd.typ}'.")
      }
    }
    if (!invalid.isEmpty) {
      fatal(
        "VCF does not support the type(s) for the following FORMAT field(s):\n" + invalid.mkString(
          "\n"
        )
      )
    }
  }

  def getAttributes(k1: String, attributes: Option[VCFMetadata]): Option[VCFAttributes] =
    attributes.flatMap(_.get(k1))

  def getAttributes(k1: String, k2: String, attributes: Option[VCFMetadata])
    : Option[VCFFieldAttributes] =
    getAttributes(k1, attributes).flatMap(_.get(k2))

  def makeHeader(
    rowType: TStruct,
    entryType: TStruct,
    rg: ReferenceGenome,
    append: Option[String],
    metadata: Option[VCFMetadata],
    sampleIds: Array[String],
  ): String = {
    val sb = new StringBuilder()

    sb.append("##fileformat=VCFv4.2\n")
    sb.append(s"##hailversion=${hail.HAIL_PRETTY_VERSION}\n")

    entryType.fields.foreach { f =>
      val attrs = getAttributes("format", f.name, metadata).getOrElse(Map.empty[String, String])
      sb.append("##FORMAT=<ID=")
      sb.append(f.name)
      sb.append(",Number=")
      sb.append(attrs.getOrElse("Number", infoNumber(f.typ)))
      sb.append(",Type=")
      sb.append(formatType(f.name, f.typ))
      sb.append(",Description=\"")
      sb.append(attrs.getOrElse("Description", "").replace("\\", "\\\\").replace("\"", "\\\""))
      sb.append("\">\n")
    }

    val filters =
      getAttributes("filter", metadata).getOrElse(Map.empty[String, Any]).keys.toArray.sorted
    filters.foreach { id =>
      val attrs = getAttributes("filter", id, metadata).getOrElse(Map.empty[String, String])
      sb.append("##FILTER=<ID=")
      sb.append(id)
      sb.append(",Description=\"")
      sb.append(attrs.getOrElse("Description", "").replace("\\", "\\\\").replace("\"", "\\\""))
      sb.append("\">\n")
    }

    val tinfo = rowType.selfField("info") match {
      case Some(fld) if fld.typ.isInstanceOf[TStruct] =>
        fld.typ.asInstanceOf[TStruct]
      case _ =>
        TStruct()
    }

    tinfo.fields.foreach { f =>
      val attrs = getAttributes("info", f.name, metadata).getOrElse(Map.empty[String, String])
      sb.append("##INFO=<ID=")
      sb.append(f.name)
      sb.append(",Number=")
      sb.append(attrs.getOrElse("Number", infoNumber(f.typ)))
      sb.append(",Type=")
      sb.append(infoType(f))
      sb.append(",Description=\"")
      sb.append(attrs.getOrElse("Description", "").replace("\\", "\\\\").replace("\"", "\\\""))
      sb.append("\">\n")
    }

    append.foreach(append => sb.append(append))

    val assembly = rg.name
    rg.contigs.foreachBetween { c =>
      sb.append("##contig=<ID=")
      sb.append(c)
      sb.append(",length=")
      sb.append(rg.contigLength(c))
      sb.append(",assembly=")
      sb.append(assembly)
      sb += '>'
    }(sb += '\n')

    sb += '\n'

    sb.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO")
    if (sampleIds.nonEmpty) {
      sb.append("\tFORMAT")
      sampleIds.foreach { id =>
        sb += '\t'
        sb.append(id)
      }
    }
    sb.result()
  }

  def lookupVAField(
    rowType: TStruct,
    fieldName: String,
    vcfColName: String,
    expectedTypeOpt: Option[Type],
  ): (Boolean, Int) = {
    rowType.fieldIdx.get(fieldName) match {
      case Some(idx) =>
        val t = rowType.types(idx)
        if (expectedTypeOpt.forall(t == _)) // FIXME: make sure this is right
          (true, idx)
        else {
          warn(s"export_vcf found row field $fieldName with type '$t', but expected type ${expectedTypeOpt.get}. " +
            s"Emitting missing $vcfColName.")
          (false, 0)
        }
      case None => (false, 0)
    }
  }
}

object TabixVCF {
  def apply(fs: FS, filePath: String): Unit = {
    val idx = using(new BGzipLineReader(fs, filePath)) { lines =>
      val tabix = new TabixIndexCreator(TabixFormat.VCF)
      var fileOffset = lines.getVirtualOffset
      var s = lines.readLine()
      while (s != null) {
        if (s.nonEmpty && s.charAt(0) != '#') {
          val Array(chrom, posStr, _*) = s.split("\t", 3)
          val pos = posStr.toInt
          val feature = new SimpleFeature(chrom, pos, pos)
          tabix.addFeature(feature, fileOffset)
        }

        fileOffset = lines.getVirtualOffset
        s = lines.readLine()
      }

      tabix.finalizeIndex(fileOffset)
    }
    val tabixPath =
      htsjdk.tribble.util.ParsingUtils.appendToPath(filePath, FileExtensions.TABIX_INDEX)
    using(new BGzipOutputStream(fs.createNoCompression(tabixPath))) { bgzos =>
      using(new htsjdk.tribble.util.LittleEndianOutputStream(bgzos))(os => idx.write(os))
    }
  }
}

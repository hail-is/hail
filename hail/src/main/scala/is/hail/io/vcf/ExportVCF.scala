package is.hail.io.vcf

import htsjdk.samtools.util.FileExtensions
import htsjdk.tribble.SimpleFeature
import htsjdk.tribble.index.tabix.{TabixFormat, TabixIndexCreator}
import is.hail
import is.hail.annotations.Region
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.MatrixValue
import is.hail.io.compress.{BGzipLineReader, BGzipOutputStream}
import is.hail.io.fs.FS
import is.hail.io.{VCFAttributes, VCFFieldAttributes, VCFMetadata}
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.variant.{Call, RegionValueVariant}

import scala.io.Source

object ExportVCF {
  def infoNumber(t: Type): String = t match {
    case TBoolean => "0"
    case TArray(_) => "."
    case TSet(_) => "."
    case _ => "1"
  }

  def strVCF(sb: StringBuilder, elementType: PType, offset: Long) {
    elementType match {
      case PInt32(_) =>
        val x = Region.loadInt(offset)
        sb.append(x)
      case PInt64(_) =>
        val x = Region.loadLong(offset)
        if (x > Int.MaxValue || x < Int.MinValue)
          fatal(s"Cannot convert Long to Int if value is greater than Int.MaxValue (2^31 - 1) " +
            s"or less than Int.MinValue (-2^31). Found $x.")
        sb.append(x)
      case PFloat32(_) =>
        val x = Region.loadFloat(offset)
        if (x.isNaN)
          sb += '.'
        else
          sb.append(x.formatted("%.6g"))
      case PFloat64(_) =>
        val x = Region.loadDouble(offset)
        if (x.isNaN)
          sb += '.'
        else
          sb.append(x.formatted("%.6g"))
      case t@PCanonicalString(_) =>
        sb.append(t.loadString(offset))
      case _: PCall =>
        val c = Region.loadInt(offset)
        Call.vcfString(c, sb)
      case _ =>
        fatal(s"VCF does not support type $elementType")
    }
  }

  def iterableVCF(sb: StringBuilder, t: PContainer, length: Int, offset: Long, delim: Char) {
    if (length > 0) {
      var i = 0
      while (i < length) {
        if (i > 0)
          sb += delim
        if (t.isElementDefined(offset, i)) {
          val eOffset = t.loadElement(offset, length, i)
          strVCF(sb, t.elementType, eOffset)
        } else
          sb += '.'
        i += 1
      }
    } else
      sb += '.'
  }

  def emitInfo(sb: StringBuilder, f: PField, offset: Long, wroteLast: Boolean): Boolean = {
    f.typ match {
      case it: PContainer if it.elementType.virtualType != TBoolean =>
        val length = it.loadLength(offset)
        if (length == 0)
          wroteLast
        else {
          if (wroteLast)
            sb += ';'
          sb.append(f.name)
          sb += '='
          iterableVCF(sb, it, length, offset, ',')
          true
        }
      case PBoolean(_) =>
        if (Region.loadBoolean(offset)) {
          if (wroteLast)
            sb += ';'
          sb.append(f.name)
          true
        } else
          wroteLast
      case t =>
        if (wroteLast)
          sb += ';'
        sb.append(f.name)
        sb += '='
        strVCF(sb, t, offset)
        true
    }
  }

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
      case _ => fatal(s"INFO field '${ f.name }': VCF does not support type '${ f.typ }'.")
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

  def checkFormatSignature(tg: TStruct) {
    tg.fields.foreach { fd =>
      val valid = fd.typ match {
        case it: TContainer => validFormatType(it.elementType)
        case t => validFormatType(t)
      }
      if (!valid)
        fatal(s"Invalid type for format field '${ fd.name }'. Found '${ fd.typ }'.")
    }
  }

  def emitGenotype(sb: StringBuilder, formatFieldOrder: Array[Int], tg: PStruct, offset: Long, fieldDefined: Array[Boolean], missingFormat: String) {
    var i = 0
    while (i < formatFieldOrder.length) {
      fieldDefined(i) = tg.isFieldDefined(offset, formatFieldOrder(i))
      i += 1
    }

    var end = i
    while (end > 0 && !fieldDefined(end - 1))
      end -= 1

    if (end == 0)
      sb.append(missingFormat)
    else {
      i = 0
      while (i < end) {
        if (i > 0)
          sb += ':'
        val j = formatFieldOrder(i)
        val fIsDefined = fieldDefined(i)
        val fOffset = tg.loadField(offset, j)

        tg.fields(j).typ match {
          case it: PContainer =>
            val pt = it
            if (fIsDefined) {
              val fLength = pt.loadLength(fOffset)
              iterableVCF(sb, pt, fLength, fOffset, ',')
            } else
              sb += '.'
          case t =>
            if (fIsDefined)
              strVCF(sb, t, fOffset)
            else if (t.virtualType == TCall)
              sb.append("./.")
            else
              sb += '.'
        }
        i += 1
      }
    }
  }

  def getAttributes(k1: String, attributes: Option[VCFMetadata]): Option[VCFAttributes] =
    attributes.flatMap(_.get(k1))

  def getAttributes(k1: String, k2: String, attributes: Option[VCFMetadata]): Option[VCFFieldAttributes] =
    getAttributes(k1, attributes).flatMap(_.get(k2))

  def apply(ctx: ExecuteContext, mv: MatrixValue, path: String, append: Option[String],
    exportType: String, metadata: Option[VCFMetadata], tabix: Boolean = false) {
    val fs = ctx.fs

    mv.typ.requireColKeyString()
    mv.typ.requireRowKeyVariant()

    val typ = mv.typ

    val tg = mv.entryPType

    checkFormatSignature(tg.virtualType)

    val formatFieldOrder: Array[Int] = tg.fieldIdx.get("GT") match {
      case Some(i) => (i +: tg.fields.filter(fd => fd.name != "GT").map(_.index)).toArray
      case None => tg.fields.indices.toArray
    }
    val formatFieldString = formatFieldOrder.map(i => tg.fields(i).name).mkString(":")

    val missingFormatStr = if (typ.entryType.size > 0 && typ.entryType.types(formatFieldOrder(0)) == TCall)
      "./."
    else "."

    val tinfo =
      if (typ.rowType.hasField("info")) {
        typ.rowType.field("info").typ match {
          case _: TStruct => mv.rvRowPType.field("info").typ.asInstanceOf[PStruct]
          case t =>
            warn(s"export_vcf found row field 'info' of type $t, but expected type 'Struct'. Emitting no INFO fields.")
            PCanonicalStruct.empty()
        }
      } else {
        warn(s"export_vcf found no row field 'info'. Emitting no INFO fields.")
        PCanonicalStruct.empty()
      }

    val rg = mv.referenceGenome
    val assembly = rg.name

    val localNSamples = mv.nCols
    val hasSamples = localNSamples > 0

    def header: String = {
      val sb = new StringBuilder()

      sb.append("##fileformat=VCFv4.2\n")
      sb.append(s"##hailversion=${ hail.HAIL_PRETTY_VERSION }\n")

      tg.fields.foreach { f =>
        val attrs = getAttributes("format", f.name, metadata).getOrElse(Map.empty[String, String])
        sb.append("##FORMAT=<ID=")
        sb.append(f.name)
        sb.append(",Number=")
        sb.append(attrs.getOrElse("Number", infoNumber(f.typ.virtualType)))
        sb.append(",Type=")
        sb.append(formatType(f.name, f.typ.virtualType))
        sb.append(",Description=\"")
        sb.append(attrs.getOrElse("Description", ""))
        sb.append("\">\n")
      }

      val filters = getAttributes("filter", metadata).getOrElse(Map.empty[String, Any]).keys.toArray.sorted
      filters.foreach { id =>
        val attrs = getAttributes("filter", id, metadata).getOrElse(Map.empty[String, String])
        sb.append("##FILTER=<ID=")
        sb.append(id)
        sb.append(",Description=\"")
        sb.append(attrs.getOrElse("Description", ""))
        sb.append("\">\n")
      }

      tinfo.virtualType.fields.foreach { f =>
        val attrs = getAttributes("info", f.name, metadata).getOrElse(Map.empty[String, String])
        sb.append("##INFO=<ID=")
        sb.append(f.name)
        sb.append(",Number=")
        sb.append(attrs.getOrElse("Number", infoNumber(f.typ)))
        sb.append(",Type=")
        sb.append(infoType(f))
        sb.append(",Description=\"")
        sb.append(attrs.getOrElse("Description", ""))
        sb.append("\">\n")
      }

      append.foreach { f =>
        using(fs.open(f)) { s =>
          Source.fromInputStream(s)
            .getLines()
            .filterNot(_.isEmpty)
            .foreach { line =>
              sb.append(line)
              sb += '\n'
            }
        }
      }

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
      if (hasSamples)
        sb.append("\tFORMAT")
      mv.stringSampleIds.foreach { id =>
        sb += '\t'
        sb.append(id)
      }
      sb.result()
    }

    val fieldIdx = typ.rowType.fieldIdx

    def lookupVAField(fieldName: String, vcfColName: String, expectedTypeOpt: Option[Type]): (Boolean, Int) = {
      fieldIdx.get(fieldName) match {
        case Some(idx) =>
          val t = typ.rowType.types(idx)
          if (expectedTypeOpt.forall(t == _)) // FIXME: make sure this is right
            (true, idx)
          else {
            warn(s"export_vcf found row field $fieldName with type '$t', but expected type ${ expectedTypeOpt.get }. " +
              s"Emitting missing $vcfColName.")
            (false, 0)
          }
        case None => (false, 0)
      }
    }
    val filtersType = TSet(TString)
    val filtersPType = if (typ.rowType.hasField("filters")) {
      assert(typ.rowType.fieldType("filters") == TSet(TString))
      mv.rvRowPType.field("filters").typ.asInstanceOf[PSet]
    } else null

    val (idExists, idIdx) = lookupVAField("rsid", "ID", Some(TString))
    val (qualExists, qualIdx) = lookupVAField("qual", "QUAL", Some(TFloat64))
    val (filtersExists, filtersIdx) = lookupVAField("filters", "FILTERS", Some(filtersType))
    val (infoExists, infoIdx) = lookupVAField("info", "INFO", None)

    val fullRowType = mv.rvRowPType
    val localEntriesIndex = mv.entriesIdx
    val localEntriesType = mv.entryArrayPType

    mv.rvd.mapPartitions { (_, it) =>
      val sb = new StringBuilder

      val formatDefinedArray = new Array[Boolean](formatFieldOrder.length)

      val rvv = new RegionValueVariant(fullRowType)
      it.map { ptr =>
        sb.clear()

        rvv.set(ptr)

        sb.append(rvv.contig())
        sb += '\t'
        sb.append(rvv.position())
        sb += '\t'

        if (idExists && fullRowType.isFieldDefined(ptr, idIdx)) {
          val idOffset = fullRowType.loadField(ptr, idIdx)
          sb.append(fullRowType.types(idIdx).asInstanceOf[PString].loadString(idOffset))
        } else
          sb += '.'

        sb += '\t'
        sb.append(rvv.alleles()(0))
        sb += '\t'
        if (rvv.alleles().length > 1) {
          rvv.alleles().tail.foreachBetween(aa =>
            sb.append(aa))(sb += ',')
        } else {
          sb += '.'
        }
        sb += '\t'

        if (qualExists && fullRowType.isFieldDefined(ptr, qualIdx)) {
          val qualOffset = fullRowType.loadField(ptr, qualIdx)
          sb.append(Region.loadDouble(qualOffset).formatted("%.2f"))
        } else
          sb += '.'

        sb += '\t'

        if (filtersExists && fullRowType.isFieldDefined(ptr, filtersIdx)) {
          val filtersOffset = fullRowType.loadField(ptr, filtersIdx)
          val filtersLength = filtersPType.loadLength(filtersOffset)
          if (filtersLength == 0)
            sb.append("PASS")
          else
            iterableVCF(sb, filtersPType, filtersLength, filtersOffset, ';')
        } else
          sb += '.'

        sb += '\t'

        var wroteAnyInfo: Boolean = false
        if (infoExists && fullRowType.isFieldDefined(ptr, infoIdx)) {
          var wrote: Boolean = false
          val infoOffset = fullRowType.loadField(ptr, infoIdx)
          var i = 0
          while (i < tinfo.size) {
            if (tinfo.isFieldDefined(infoOffset, i)) {
              wrote = emitInfo(sb, tinfo.fields(i), tinfo.loadField(infoOffset, i), wrote)
              wroteAnyInfo = wroteAnyInfo || wrote
            }
            i += 1
          }
        }
        if (!wroteAnyInfo)
          sb += '.'

        if (hasSamples) {
          sb += '\t'
          sb.append(formatFieldString)

          val gsOffset = fullRowType.loadField(ptr, localEntriesIndex)
          var i = 0
          while (i < localNSamples) {
            sb += '\t'
            if (localEntriesType.isElementDefined(gsOffset, i))
              emitGenotype(sb, formatFieldOrder, tg, localEntriesType.loadElement(gsOffset, localNSamples, i), formatDefinedArray, missingFormatStr)
            else
              sb.append(missingFormatStr)

            i += 1
          }
        }

        sb.result()
      }
    }.writeTable(ctx, path, Some(header), exportType = exportType)

    if (tabix) {
      exportType match {
        case ExportType.CONCATENATED =>
          info(s"Writing tabix index for $path")
          TabixVCF(fs, path)
        case ExportType.PARALLEL_SEPARATE_HEADER | ExportType.PARALLEL_HEADER_IN_SHARD =>
          val files = fs.glob(path + "/part-*").map(_.getPath.getBytes)
          info(s"Writing tabix index for ${ files.length } in $path")
          ctx.backend.parallelizeAndComputeWithIndex(ctx.backendContext, ctx.fs, files)({ (pathBytes, _, _, fs) =>
                      TabixVCF(fs, new String(pathBytes))
                      Array.empty
                    })
        case ExportType.PARALLEL_COMPOSABLE =>
          warn("Writing tabix index for `parallel=composable` is not supported. No index will be written.")
      }
    }
  }
}

object TabixVCF {
   def apply(fs: FS, filePath: String) {
     val idx = using (new BGzipLineReader(fs, filePath)) { lines =>
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
     val tabixPath = htsjdk.tribble.util.ParsingUtils.appendToPath(filePath, FileExtensions.TABIX_INDEX)
     using (new BGzipOutputStream(fs.createNoCompression(tabixPath))) { bgzos =>
       using (new htsjdk.tribble.util.LittleEndianOutputStream(bgzos)) { os => idx.write(os) }
     }
   }
}

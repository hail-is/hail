package is.hail.io.vcf

import is.hail
import is.hail.annotations.Region
import is.hail.expr._
import is.hail.expr.types._
import is.hail.io.{VCFAttributes, VCFFieldAttributes, VCFMetadata}
import is.hail.utils._
import is.hail.variant.{Genotype, MatrixTable, Variant}

import scala.io.Source

object ExportVCF {
  def infoNumber(t: Type): String = t match {
    case TBoolean(_) => "0"
    case TArray(_, _) => "."
    case TSet(_, _) => "."
    case _ => "1"
  }

  def strVCF(sb: StringBuilder, elementType: Type, m: Region, offset: Long) {
    elementType match {
      case TInt32(_) =>
        val x = m.loadInt(offset)
        sb.append(x)
      case TInt64(_) =>
        val x = m.loadLong(offset)
        if (x > Int.MaxValue || x < Int.MinValue)
          fatal(s"Cannot convert Long to Int if value is greater than Int.MaxValue (2^31 - 1) " +
            s"or less than Int.MinValue (-2^31). Found $x.")
        sb.append(x)
      case TFloat32(_) =>
        val x = m.loadFloat(offset)
        if (x.isNaN)
          sb += '.'
        else
          sb.append(x.formatted("%.5e"))
      case TFloat64(_) =>
        val x = m.loadDouble(offset)
        if (x.isNaN)
          sb += '.'
        else
          sb.append(x.formatted("%.5e"))
      case TString(_) =>
        sb.append(TString.loadString(m, offset))
      case TCall(_) =>
        val p = Genotype.gtPair(m.loadInt(offset))
        sb.append(p.j)
        sb += '/'
        sb.append(p.k)
      case _ =>
        fatal(s"VCF does not support type $elementType")
    }
  }
  
  def iterableVCF(sb: StringBuilder, t: TIterable, m: Region, length: Int, offset: Long) {
    if (length > 0) {
      var i = 0
      while (i < length) {
        if (i > 0)
          sb += ','
        if (t.isElementDefined(m, offset, i)) {
          val eOffset = t.loadElement(m, offset, length, i)
          strVCF(sb, t.elementType, m, eOffset)
        } else
          sb += '.'
        i += 1
      }
    } else
      sb += '.'
  }

  def emitInfo(sb: StringBuilder, f: Field, m: Region, offset: Long, wroteLast: Boolean): Boolean = {
    f.typ match {
      case it: TIterable if !it.elementType.isOfType(TBoolean()) =>
        val length = it.loadLength(m, offset)
        if (length == 0)
          wroteLast
        else {
          if (wroteLast)
            sb += ';'
          sb.append(f.name)
          sb += '='
          iterableVCF(sb, it, m, length, offset)
          true
        }
      case TBoolean(_) =>
        if (m.loadBoolean(offset)) {
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
        strVCF(sb, t, m, offset)
        true
    }
  }

  def infoType(t: Type): Option[String] = t match {
    case _: TInt32 | _: TInt64 => Some("Integer")
    case _: TFloat64 | _: TFloat32 => Some("Float")
    case _: TString => Some("String")
    case _: TBoolean => Some("Flag")
    case _ => None
  }

  def infoType(f: Field): String = {
    val tOption = f.typ match {
      case TArray(TBoolean(_), _) | TSet(TBoolean(_), _) => None
      case TArray(elt, _) => infoType(elt)
      case TSet(elt, _) => infoType(elt)
      case t => infoType(t)
    }
    tOption match {
      case Some(s) => s
      case _ => fatal(s"INFO field '${ f.name }': VCF does not support type `${ f.typ }'.")
    }
  }

  def formatType(t: Type): Option[String] = t match {
    case _: TInt32 | _: TInt64 => Some("Integer")
    case _: TFloat64 | _: TFloat32 => Some("Float")
    case _: TString => Some("String")
    case _: TCall => Some("String")
    case _ => None
  }

  def formatType(f: Field): String = {
    val tOption = f.typ match {
      case TArray(elt, _) => formatType(elt)
      case TSet(elt, _) => formatType(elt)
      case t => formatType(t)
    }

    tOption match {
      case Some(s) => s
      case _ => fatal(s"FORMAT field '${ f.name }': VCF does not support type `${ f.typ }'.")
    }
  }

  def validFormatType(typ: Type): Boolean = {
    typ match {
      case _: TString => true
      case _: TFloat64 => true
      case _: TFloat32 => true
      case _: TInt32 => true
      case _: TInt64 => true
      case _: TCall => true
      case _ => false
    }
  }
  
  def checkFormatSignature(tg: TStruct) {
    tg.fields.foreach { fd =>
      val valid = fd.typ match {
        case it: TIterable => validFormatType(it.elementType)
        case t => validFormatType(t)
      }
      if (!valid)
        fatal(s"Invalid type for format field `${ fd.name }'. Found ${ fd.typ }.")
    }
  }
  
  def emitGenotype(sb: StringBuilder, formatFieldOrder: Array[Int], tg: TStruct, m: Region, offset: Long) {
    formatFieldOrder.foreachBetween { j =>
      val fIsDefined = tg.isFieldDefined(m, offset, j)
      val fOffset = tg.loadField(m, offset, j)

      tg.fields(j).typ match {
        case it: TIterable =>
          if (fIsDefined) {
            val fLength = it.loadLength(m, fOffset)
            iterableVCF(sb, it, m, fLength, fOffset)
          } else
            sb += '.'
        case t =>
          if (fIsDefined)
            strVCF(sb, t, m, fOffset)
          else if (t.isOfType(TCall()))
            sb.append("./.")
          else
            sb += '.'
      }
    }(sb += ':')
  }

  def getAttributes(k1: String, attributes: Option[VCFMetadata]): Option[VCFAttributes] =
    attributes.flatMap(_.get(k1))

  def getAttributes(k1: String, k2: String, attributes: Option[VCFMetadata]): Option[VCFFieldAttributes] =
    getAttributes(k1, attributes).flatMap(_.get(k2))

  def getAttributes(k1: String, k2: String, k3: String, attributes: Option[VCFMetadata]): Option[String] =
    getAttributes(k1, k2, attributes).flatMap(_.get(k3))

  def apply(vsm: MatrixTable, path: String, append: Option[String] = None,
    exportType: Int = ExportType.CONCATENATED, metadata: Option[VCFMetadata] = None) {
    
    vsm.requireColKeyString("export_vcf")
    vsm.requireRowKeyVariant("export_vcf")
    
    val tg = vsm.genotypeSignature match {
      case t: TStruct => t
      case t =>
        fatal(s"export_vcf requires g to have type TStruct, found $t")
    }

    checkFormatSignature(tg)
        
    val formatFieldOrder: Array[Int] = tg.fieldIdx.get("GT") match {
      case Some(i) => (i +: tg.fields.filter(fd => fd.name != "GT").map(_.index)).toArray
      case None => tg.fields.indices.toArray
    }
    val formatFieldString = formatFieldOrder.map(i => tg.fields(i).name).mkString(":")

    val tva = vsm.vaSignature match {
      case t: TStruct => t.asInstanceOf[TStruct]
      case _ =>
        warn(s"export_vcf found va of type ${ vsm.vaSignature }, but expected type TStruct. " +
          "Emitting missing RSID, QUAL, and INFO.")
        TStruct.empty()
    }
    
    val tinfo =
      if (tva.hasField("info")) {
        tva.field("info").typ match {
          case t: TStruct => t.asInstanceOf[TStruct]
          case t =>
            warn(s"export_vcf found va.info of type $t, but expected type TStruct. Emitting missing INFO.")
            TStruct.empty()
        }
      } else
        TStruct.empty()
    
    val gr = vsm.genomeReference
    val assembly = gr.name
    
    val localNSamples = vsm.nSamples
    val hasSamples = localNSamples > 0

    def header: String = {
      val sb = new StringBuilder()

      sb.append("##fileformat=VCFv4.2\n")
      sb.append(s"##hailversion=${ hail.HAIL_PRETTY_VERSION }\n")

      tg.fields.foreachBetween { f =>
        val attrs = getAttributes("format", f.name, metadata).getOrElse(Map.empty[String, String])
        sb.append("##FORMAT=<ID=")
        sb.append(f.name)
        sb.append(",Number=")
        sb.append(attrs.getOrElse("Number", infoNumber(f.typ)))
        sb.append(",Type=")
        sb.append(formatType(f))
        sb.append(",Description=\"")
        sb.append(attrs.getOrElse("Description", ""))
        sb.append("\">")
      }(sb += '\n')

      sb += '\n'

      val filters = getAttributes("filter", metadata).getOrElse(Map.empty[String, Any]).keys.toArray.sorted
      filters.foreach { id =>
        val attrs = getAttributes("filter", id, metadata).getOrElse(Map.empty[String, String])
        sb.append("##FILTER=<ID=")
        sb.append(id)
        sb.append(",Description=\"")
        sb.append(attrs.getOrElse("Description", ""))
        sb.append("\">\n")
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
        sb.append(attrs.getOrElse("Description", ""))
        sb.append("\">\n")
      }

      append.foreach { f =>
        vsm.sparkContext.hadoopConfiguration.readFile(f) { s =>
          Source.fromInputStream(s)
            .getLines()
            .filterNot(_.isEmpty)
            .foreach { line =>
              sb.append(line)
              sb += '\n'
            }
        }
      }

      gr.contigs.foreachBetween { c =>
        sb.append("##contig=<ID=")
        sb.append(c)
        sb.append(",length=")
        sb.append(gr.contigLength(c))
        sb.append(",assembly=")
        sb.append(assembly)
        sb += '>'
      }(sb += '\n')

      sb += '\n'

      sb.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO")
      if (hasSamples)
        sb.append("\tFORMAT")
      vsm.sampleIds.foreach { id =>
        sb += '\t'
        sb.append(id)
      }
      sb.result()
    }
    
    val fieldIdx = tva.fieldIdx
    
    def lookupVAField(fieldName: String, vcfColName: String, expectedTypeOpt: Option[Type]): (Boolean, Int) = {
      fieldIdx.get(fieldName) match {
        case Some(idx) =>
          val t = tva.fields(idx).typ
          if (expectedTypeOpt.forall(t == _)) // FIXME: make sure this is right
            (true, idx)
          else {
            warn(s"export_vcf found va.$fieldName with type `$t', but expected type ${ expectedTypeOpt.get }. " +
              s"Emitting missing $vcfColName.")
            (false, 0)
          }
        case None => (false, 0)
      }
    }
    
    val (idExists, idIdx) = lookupVAField("rsid", "ID", Some(TString()))
    val (qualExists, qualIdx) = lookupVAField("qual", "QUAL", Some(TFloat64()))
    val (filtersExists, filtersIdx) = lookupVAField("filters", "FILTERS", Some(TSet(TString())))
    val (infoExists, infoIdx) = lookupVAField("info", "INFO", None)
    
    val localRowType = vsm.rowType
    val tgs = localRowType.fields(3).typ.asInstanceOf[TArray]
    
    vsm.rdd2.mapPartitions { it =>
      val sb = new StringBuilder
      var m: Region = null
      
      it.map { rv =>
        sb.clear()

        m = rv.region
        
        val vOffset = localRowType.loadField(m, rv.offset, 1)
        val vaOffset = localRowType.loadField(m, rv.offset, 2)
        val gsOffset = localRowType.loadField(m, rv.offset, 3)
        
        val v = Variant.fromRegionValue(m, vOffset)
        
        sb.append(v.contig)
        sb += '\t'
        sb.append(v.start)
        sb += '\t'
  
        if (idExists && tva.isFieldDefined(m, vaOffset, idIdx)) {
          val idOffset = tva.loadField(m, vaOffset, idIdx)
          sb.append(TString.loadString(m, idOffset))
        } else
          sb += '.'
  
        sb += '\t'
        sb.append(v.ref)
        sb += '\t'
        v.altAlleles.foreachBetween(aa =>
          sb.append(aa.alt))(sb += ',')
        sb += '\t'

        if (qualExists && tva.isFieldDefined(m, vaOffset, qualIdx)) {
          val qualOffset = tva.loadField(m, vaOffset, qualIdx)
          sb.append(m.loadDouble(qualOffset).formatted("%.2f"))
        } else
          sb += '.'
        
        sb += '\t'
        
        if (filtersExists && tva.isFieldDefined(m, vaOffset, filtersIdx)) {
          val filtersOffset = tva.loadField(m, vaOffset, filtersIdx)
          val filtersLength = TSet(TString()).loadLength(m, filtersOffset)
          if (filtersLength == 0)
            sb.append("PASS")
          else
            iterableVCF(sb, TSet(TString()), m, filtersLength, filtersOffset)
        } else
          sb += '.'
  
        sb += '\t'
        
        var wroteAnyInfo: Boolean = false
        if (infoExists && tva.isFieldDefined(m, vaOffset, infoIdx)) {
          var wrote: Boolean = false
          val infoOffset = tva.loadField(m, vaOffset, infoIdx)          
          var i = 0
          while (i < tinfo.size) {
            if (tinfo.isFieldDefined(m, infoOffset, i)) {
              wrote = emitInfo(sb, tinfo.fields(i), m, tinfo.loadField(m, infoOffset, i), wrote)
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
          
          var i = 0
          while (i < localNSamples) {
            sb += '\t'
            if (tgs.isElementDefined(m, gsOffset, i))
              emitGenotype(sb, formatFieldOrder, tg, m, tgs.loadElement(m, gsOffset, localNSamples, i))
            else
              sb.append("./.")

            i += 1
          }
        }
        
        sb.result()
      }
    }.writeTable(path, vsm.hc.tmpDir, Some(header), exportType = exportType)
  }
}

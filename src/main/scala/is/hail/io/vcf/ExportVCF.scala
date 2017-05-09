package is.hail.io.vcf

import is.hail.annotations.{Annotation, Querier}
import is.hail.expr.{Field, TArray, TBoolean, TCall, TDouble, TFloat, TGenotype, TInt, TIterable, TLong, TSet, TString, TStruct, Type}
import is.hail.utils._
import is.hail.variant.{Call, GenericDataset, GenomeReference, Genotype, Variant}
import org.apache.spark.sql.Row

import scala.io.Source

object ExportVCF {

  def infoNumber(t: Type): String = t match {
    case TBoolean => "0"
    case TArray(elementType) => "."
    case _ => "1"
  }

  def strVCF(sb: StringBuilder, elementType: Type, a: Annotation, gr: GenomeReference) {
    if (a == null)
      sb += '.'
    else {
      elementType match {
        case TFloat =>
          val x = a.asInstanceOf[Float]
          if (x.isNaN)
            sb += '.'
          else
            sb.append(x.formatted("%.5e"))
        case TDouble =>
          val x = a.asInstanceOf[Double]
          if (x.isNaN)
            sb += '.'
          else
            sb.append(x.formatted("%.5e"))
        case TLong =>
          val x = a.asInstanceOf[Long]
          if (x > Int.MaxValue || x < Int.MinValue)
            fatal(s"Cannot convert Long to Int if value is greater than Int.MaxValue (2^31 - 1) or less than Int.MinValue (-2^31). Found $x.")
          sb.append(elementType.str(x, gr))
        case _ => sb.append(elementType.str(a, gr))
      }
    }
  }

  def emitFormatField(f: Field, sb: StringBuilder, a: Annotation, gr: GenomeReference) {
    f.typ match {
      case TCall => sb.append(Call.toString(a.asInstanceOf[Call]))
      case it: TIterable =>
        if (a == null)
          sb += '.'
        else {
          val arr = a.asInstanceOf[Iterable[_]]
          arr.foreachBetween(a => strVCF(sb, it.elementType, a, gr))(sb += ',')
        }
      case t => strVCF(sb, t, a, gr)
    }
  }

  def emitInfo(f: Field, sb: StringBuilder, value: Annotation, gr: GenomeReference): Boolean = {
    if (value == null)
      false
    else
      f.typ match {
        case it: TIterable =>
          val arr = value.asInstanceOf[Iterable[_]]
          if (arr.isEmpty) {
            false // missing and empty iterables treated the same
          } else {
            sb.append(f.name)
            sb += '='
            arr.foreachBetween(a => strVCF(sb, it.elementType, a, gr))(sb += ',')
            true
          }
        case TBoolean => value match {
          case true =>
            sb.append(f.name)
            true
          case _ =>
            false
        }
        case t =>
          sb.append(f.name)
          sb += '='
          strVCF(sb, t, value, gr)
          true
      }
  }

  def infoType(t: Type): Option[String] = t match {
    case TInt | TLong => Some("Integer")
    case TDouble | TFloat => Some("Float")
    case TString => Some("String")
    case TBoolean => Some("Flag")
    case _ => None
  }

  def infoType(f: Field): String = {
    val tOption = f.typ match {
      case TArray(elt) => infoType(elt)
      case TSet(elt) => infoType(elt)
      case t => infoType(t)
    }
    tOption match {
      case Some(s) => s
      case _ => fatal(s"INFO field '${ f.name }': VCF does not support type `${ f.typ }'.")
    }
  }

  def formatType(t: Type): Option[String] = t match {
    case TInt | TLong => Some("Integer")
    case TDouble | TFloat => Some("Float")
    case TString => Some("String")
    case TCall => Some("String")
    case _ => None
  }

  def formatType(f: Field): String = {
    val tOption = f.typ match {
      case TArray(elt) => formatType(elt)
      case TSet(elt) => formatType(elt)
      case t => formatType(t)
    }

    tOption match {
      case Some(s) => s
      case _ => fatal(s"FORMAT field '${ f.name }': VCF does not support type `${ f.typ }'.")
    }
  }

  def appendIntArrayOption(sb: StringBuilder, toAppend: Option[Array[Int]]): Unit = {
    toAppend match {
      case Some(i) => i.foreachBetween(sb.append(_))(sb += ',')
      case None => sb += '.'
    }
  }

  def appendIntOption(sb: StringBuilder, toAppend: Option[Int]): Unit = {
    toAppend match {
      case Some(i) => sb.append(i)
      case None => sb += '.'
    }
  }

  def appendIntArray(sb: StringBuilder, toAppend: Array[Int]): Unit = {
    toAppend.foreachBetween(sb.append(_))(sb += ',')
  }

  def appendDoubleArray(sb: StringBuilder, toAppend: Array[Double]): Unit = {
    toAppend.foreachBetween(sb.append(_))(sb += ',')
  }

  def writeGenotype(sb: StringBuilder, sig: TStruct, fieldOrder: Array[Int], a: Annotation, gr: GenomeReference) {
    val r = a.asInstanceOf[Row]
    val fields = sig.fields
    assert(r.length == fields.length, "annotation/type mismatch")

    fieldOrder.foreachBetween { i =>
      emitFormatField(fields(i), sb, r.get(i), gr)
    }(sb += ':')
  }

  def writeGenotype(sb: StringBuilder, g: Genotype) {
    sb.append(g.gt.map { gt =>
      val p = Genotype.gtPair(gt)
      s"${ p.j }/${ p.k }"
    }.getOrElse("./."))

    (g.ad, g.dp, g.gq,
      if (g.isLinearScale)
        g.gp.map(Left(_))
      else
        g.pl.map(Right(_))) match {
      case (None, None, None, None) =>
      case (Some(ad), None, None, None) =>
        sb += ':'
        appendIntArray(sb, ad)
      case (ad, Some(dp), None, None) =>
        sb += ':'
        appendIntArrayOption(sb, ad)
        sb += ':'
        sb.append(dp)
      case (ad, dp, Some(gq), None) =>
        sb += ':'
        appendIntArrayOption(sb, ad)
        sb += ':'
        appendIntOption(sb, dp)
        sb += ':'
        sb.append(gq)
      case (ad, dp, gq, Some(gpOrPL)) =>
        sb += ':'
        appendIntArrayOption(sb, ad)
        sb += ':'
        appendIntOption(sb, dp)
        sb += ':'
        appendIntOption(sb, gq)
        sb += ':'
        gpOrPL match {
          case Left(gp) => appendDoubleArray(sb, gp)
          case Right(pl) => appendIntArray(sb, pl)
        }
    }
  }

  def validFormatType(typ: Type): Boolean = {
    typ match {
      case TString => true
      case TDouble => true
      case TFloat => true
      case TInt => true
      case TLong => true
      case TCall => true
      case _ => false
    }
  }

  def checkFormatSignature(sig: TStruct) {
    sig.fields.foreach { fd =>
      val valid = fd.typ match {
        case it: TIterable => validFormatType(it.elementType)
        case t => validFormatType(t)
      }
      if (!valid)
        fatal(s"Invalid type for format field `${ fd.name }'. Found ${ fd.typ }.")
    }
  }

  def apply(gds: GenericDataset, path: String, append: Option[String] = None, exportPP: Boolean = false,
    parallel: Boolean = false) {
    val vas = gds.vaSignature

    val genotypeSignature = gds.genotypeSignature

    val (genotypeFormatField, genotypeFieldOrder) = genotypeSignature match {
      case TGenotype =>
        if (exportPP)
          ("GT:AD:DP:GQ:PP", null)
        else
          ("GT:AD:DP:GQ:PL", null)
      case sig: TStruct =>
        val fields = sig.fields
        val formatFieldOrder: Array[Int] = sig.fieldIdx.get("GT") match {
          case Some(i) => (i +: fields.filter(fd => fd.name != "GT").map(_.index)).toArray
          case None => sig.fields.indices.toArray
        }
        val formatFieldString = formatFieldOrder.map(i => fields(i).name).mkString(":")

        checkFormatSignature(sig)

        (formatFieldString, formatFieldOrder)
      case _ => fatal(s"Can only export to VCF with genotype signature of TGenotype or TStruct. Found `${ genotypeSignature }'.")
    }

    val infoSignature = gds.vaSignature
      .getAsOption[TStruct]("info")
    val infoQuery: (Annotation) => Option[(Annotation, TStruct)] = infoSignature.map { struct =>
      val (_, f) = gds.queryVA("va.info")
      (a: Annotation) => {
        if (a == null)
          None
        else
          Some((f(a), struct))
      }
    }.getOrElse((a: Annotation) => None)

    val hasSamples = gds.nSamples > 0

    def header: String = {
      val sb = new StringBuilder()

      sb.append("##fileformat=VCFv4.2\n")
      // FIXME add Hail version

      genotypeSignature match {
        case TGenotype =>
          if (exportPP)
            sb.append(
              """##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
              |##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths for the ref and alt alleles in the order listed">
              |##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">
              |##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
              |##FORMAT=<ID=PP,Number=G,Type=Integer,Description="Normalized, Phred-scaled posterior probabilities for genotypes as defined in the VCF specification">""".stripMargin)
          else
            sb.append(
              """##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
              |##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths for the ref and alt alleles in the order listed">
              |##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">
              |##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
              |##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Normalized, Phred-scaled likelihoods for genotypes as defined in the VCF specification">""".stripMargin)
        case sig: TStruct =>
          sig.fields.foreachBetween { f =>
            sb.append("##FORMAT=<ID=")
            sb.append(f.name)
            sb.append(",Number=")
            sb.append(f.attr("Number").getOrElse(infoNumber(f.typ)))
            sb.append(",Type=")
            sb.append(formatType(f))
            sb.append(",Description=\"")
            sb.append(f.attr("Description").getOrElse(""))
            sb.append("\">")
          }(sb += '\n')
      }

      sb += '\n'

      gds.vaSignature.fieldOption("filters")
        .foreach { f =>
          f.attrs.foreach { case (key, desc) =>
            sb.append(s"""##FILTER=<ID=$key,Description="$desc">\n""")
          }
        }

      infoSignature.foreach(_.fields.foreach { f =>
        sb.append("##INFO=<ID=")
        sb.append(f.name)
        sb.append(",Number=")
        sb.append(f.attr("Number").getOrElse(infoNumber(f.typ)))
        sb.append(",Type=")
        sb.append(infoType(f))
        sb.append(",Description=\"")
        sb.append(f.attr("Description").getOrElse(""))
        sb.append("\">\n")
      })

      append.foreach { f =>
        gds.sparkContext.hadoopConfiguration.readFile(f) { s =>
          Source.fromInputStream(s)
            .getLines()
            .filterNot(_.isEmpty)
            .foreach { line =>
              sb.append(line)
              sb += '\n'
            }
        }
      }

      sb.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO")
      if (hasSamples)
        sb.append("\tFORMAT")
      gds.sampleIds.foreach { id =>
        sb += '\t'
        sb.append(id)
      }
      sb.result()
    }

    val idQuery: Option[Querier] = vas.getOption("rsid")
      .filter {
        case TString => true
        case t => warn(
          s"""found `rsid' field, but it was an unexpected type `$t'.  Emitting missing RSID.
             |  Expected ${ TString }""".stripMargin)
          false
      }.map(_ => gds.queryVA("va.rsid")._2)

    val qualQuery: Option[Querier] = vas.getOption("qual")
      .filter {
        case TDouble => true
        case t => warn(
          s"""found `qual' field, but it was an unexpected type `$t'.  Emitting missing QUAL.
             |  Expected ${ TDouble }""".stripMargin)
          false
      }.map(_ => gds.queryVA("va.qual")._2)

    val filterQuery: Option[Querier] = vas.getOption("filters")
      .filter {
        case TSet(TString) => true
        case t =>
          warn(
            s"""found `filters' field, but it was an unexpected type `$t'.  Emitting missing FILTERS.
               |  Expected ${ TSet(TString) }""".stripMargin)
          false
      }.map(_ => gds.queryVA("va.filters")._2)

    def appendRow(sb: StringBuilder, v: Variant, a: Annotation, gs: Iterable[Annotation], gr: GenomeReference) {

      sb.append(v.contigStr(gr))
      sb += '\t'
      sb.append(v.start)
      sb += '\t'

      sb.append(idQuery.flatMap(q => Option(q(a)))
        .getOrElse("."))

      sb += '\t'
      sb.append(v.ref)
      sb += '\t'
      v.altAlleles.foreachBetween(aa =>
        sb.append(aa.alt))(sb += ',')
      sb += '\t'

      sb.append(qualQuery.flatMap(q => Option(q(a)))
        .map(_.asInstanceOf[Double].formatted("%.2f"))
        .getOrElse("."))

      sb += '\t'

      filterQuery.flatMap(q => Option(q(a)))
        .map(_.asInstanceOf[Set[String]]) match {
        case Some(f) =>
          if (f.nonEmpty)
            f.foreachBetween(s => sb.append(s))(sb += ';')
          else
            sb.append("PASS")
        case None => sb += '.'
      }

      sb += '\t'

      var wroteAnyInfo: Boolean = false
      infoQuery(a).foreach { case (anno, struct) =>
        val r = anno.asInstanceOf[Row]
        val fields = struct.fields
        assert(r.length == fields.length, "annotation/type mismatch")

        var wrote: Boolean = false
        fields.indices.foreachBetween { i =>
          wrote = emitInfo(fields(i), sb, r.get(i), gr)
          wroteAnyInfo = wroteAnyInfo || wrote
        }(if (wrote) sb += ';')
      }
      if (!wroteAnyInfo)
        sb += '.'

      if (hasSamples) {
        sb += '\t'
        sb.append(genotypeFormatField)
        gs.foreach {
          g =>
            sb += '\t'

            genotypeSignature match {
              case TGenotype => writeGenotype(sb, g.asInstanceOf[Genotype])
              case sig: TStruct => writeGenotype(sb, sig, genotypeFieldOrder, g, gr)
            }
        }
      }
    }

    val localGenomeRef = gds.genomeReference

    gds.rdd.mapPartitions { it: Iterator[(Variant, (Annotation, Iterable[Annotation]))] =>
      val sb = new StringBuilder
      it.map { case (v, (va, gs)) =>
        sb.clear()
        appendRow(sb, v, va, gs, localGenomeRef)
        sb.result()
      }
    }.writeTable(path, gds.hc.tmpDir, Some(header), parallelWrite = parallel)
  }

}

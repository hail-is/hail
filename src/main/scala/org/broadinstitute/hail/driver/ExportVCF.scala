package org.broadinstitute.hail.driver

import org.apache.spark.sql.Row
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.variant.{Genotype, Variant}
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.io.Source

object ExportVCF extends Command {

  class Options extends BaseOptions {
    @Args4jOption(name = "-a", usage = "Append file to header")
    var append: String = _

    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _

    @Args4jOption(name = "--export-pp", usage = "Export Hail PLs as a PP format field")
    var exportPP: Boolean = false

    @Args4jOption(name = "--parallel", usage = "Export VCF in parallel")
    var parallel: Boolean = false
  }

  def newOptions = new Options

  def name = "exportvcf"

  def description = "Write current dataset as VCF file"

  def supportsMultiallelic = true

  def requiresVDS = true

  def infoNumber(t: Type): String = t match {
    case TBoolean => "0"
    case TArray(elementType) => "."
    case _ => "1"
  }

  def emitInfo(f: Field, sb: StringBuilder, value: Annotation): Boolean = {
    if (value == null)
      false
    else
      f.`type` match {
        case it: TIterable =>
          val arr = value.asInstanceOf[Iterable[_]]
          if (arr.isEmpty) {
            false // missing and empty iterables treated the same
          } else {
            sb.append(f.name)
            sb += '='
            arr.foreachBetween(a => sb.append(it.elementType.strVCF(a)))(sb += ',')
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
          sb.append(t.str(value))
          true
      }
  }

  def infoType(t: Type): String = t match {
    case TArray(elementType) => infoType(elementType)
    case TInt => "Integer"
    case TDouble => "Float"
    case TChar => "Character"
    case TString => "String"
    case TBoolean => "Flag"

    // FIXME
    case _ => "String"
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

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val vas = vds.vaSignature

    val infoSignature = vds.vaSignature
      .getAsOption[TStruct]("info")
    val infoQuery: (Annotation) => Option[(Annotation, TStruct)] = infoSignature.map { struct =>
      val (_, f) = vds.queryVA("va.info")
      (a: Annotation) => f(a).map(value => (value, struct))
    }.getOrElse((a: Annotation) => None)

    val hasSamples = vds.nSamples > 0

    val exportPP = options.exportPP
    val parallel = options.parallel

    def header: String = {
      val sb = new StringBuilder()

      sb.append("##fileformat=VCFv4.2\n")
      // FIXME add Hail version
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
      sb += '\n'

      vds.vaSignature.fieldOption("filters")
        .foreach { f =>
          f.attrs.foreach { case (key, desc) =>
            sb.append(s"""##FILTER=<ID=$key,Description="$desc">\n""")
          }
        }

      infoSignature.foreach(_.fields.foreach { f =>
        sb.append("##INFO=<ID=")
        sb.append(f.name)
        sb.append(",Number=")
        sb.append(f.attr("Number").getOrElse(infoNumber(f.`type`)))
        sb.append(",Type=")
        sb.append(infoType(f.`type`))
        sb.append(",Description=\"")
        sb.append(f.attr("Description").getOrElse(""))
        sb.append("\">\n")
      })

      if (options.append != null) {
        state.hadoopConf.readFile(options.append) { s =>
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
      vds.sampleIds.foreach { id =>
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
      }.map(_ => vds.queryVA("va.rsid")._2)

    val qualQuery: Option[Querier] = vas.getOption("qual")
      .filter {
        case TDouble => true
        case t => warn(
          s"""found `qual' field, but it was an unexpected type `$t'.  Emitting missing QUAL.
              |  Expected ${ TDouble }""".stripMargin)
          false
      }.map(_ => vds.queryVA("va.qual")._2)

    val filterQuery: Option[Querier] = vas.getOption("filters")
      .filter {
        case TSet(TString) => true
        case t =>
          warn(
            s"""found `filters' field, but it was an unexpected type `$t'.  Emitting missing FILTERS.
                |  Expected ${ TSet(TString) }""".stripMargin)
          false
      }.map(_ => vds.queryVA("va.filters")._2)

    def appendRow(sb: StringBuilder, v: Variant, a: Annotation, gs: Iterable[Genotype]) {

      sb.append(v.contig)
      sb += '\t'
      sb.append(v.start)
      sb += '\t'

      sb.append(idQuery.flatMap(_ (a))
        .getOrElse("."))

      sb += '\t'
      sb.append(v.ref)
      sb += '\t'
      v.altAlleles.foreachBetween(aa =>
        sb.append(aa.alt))(sb += ',')
      sb += '\t'

      sb.append(qualQuery.flatMap(_ (a))
        .map(_.asInstanceOf[Double].formatted("%.2f"))
        .getOrElse("."))

      sb += '\t'

      filterQuery.flatMap(_ (a))
        .map(_.asInstanceOf[Set[String]]) match {
        case Some(f) =>
          if (f.nonEmpty)
            f.foreachBetween(s => sb.append(s))(sb += ';')
          else
            sb += '.'
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
          wrote = emitInfo(fields(i), sb, r.get(i))
          wroteAnyInfo = wroteAnyInfo || wrote
        }(if (wrote) sb += ';')
      }
      if (!wroteAnyInfo)
        sb += '.'

      if (hasSamples) {
        sb += '\t'
        if (exportPP)
          sb.append("GT:AD:DP:GQ:PP")
        else
          sb.append("GT:AD:DP:GQ:PL")
        gs.foreach { g =>
          sb += '\t'

          sb.append(g.gt.map { gt =>
            val p = Genotype.gtPair(gt)
            s"${ p.j }/${ p.k }"
          }.getOrElse("./."))

          (g.ad, g.dp, g.gq,
            if (g.isDosage)
              g.dosage.map(Left(_))
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
            case (ad, dp, gq, Some(dosageOrPL)) =>
              sb += ':'
              appendIntArrayOption(sb, ad)
              sb += ':'
              appendIntOption(sb, dp)
              sb += ':'
              appendIntOption(sb, gq)
              sb += ':'
              dosageOrPL match {
                case Left(dosage) => appendDoubleArray(sb, dosage)
                case Right(pl) => appendIntArray(sb, pl)
              }
          }
        }
      }
    }

    vds.rdd.mapPartitions { it: Iterator[(Variant, (Annotation, Iterable[Genotype]))] =>
      val sb = new StringBuilder
      it.map { case (v, (va, gs)) =>
        sb.clear()
        appendRow(sb, v, va, gs)
        sb.result()
      }
    }.writeTable(options.output, Some(header), parallelWrite = parallel)
    state
  }

}

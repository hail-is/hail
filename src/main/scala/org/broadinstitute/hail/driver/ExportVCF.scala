package org.broadinstitute.hail.driver

import java.time._

import org.apache.spark.RangePartitioner
import org.apache.spark.sql.Row
import org.apache.spark.storage.StorageLevel
import org.broadinstitute.hail.Utils._
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
  }

  def newOptions = new Options

  def name = "exportvcf"

  def description = "Write current dataset as VCF file"

  def supportsMultiallelic = true

  def requiresVDS = true

  def infoNumber(t: BaseType): String = t match {
    case TBoolean => "0"
    case TArray(elementType) => "."
    case _ => "1"
  }

  def infoType(t: BaseType): String = t match {
    case TArray(elementType) => infoType(elementType)
    case TInt => "Integer"
    case TDouble => "Float"
    case TChar => "Character"
    case TString => "String"
    case TBoolean => "Flag"

    // FIXME
    case _ => "String"
  }

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val vas = vds.vaSignature

    val infoSignature = vds.vaSignature
      .getAsOption[TStruct]("info")
    val infoQuery: Querier = infoSignature match {
      case Some(_) => vds.queryVA("va.info")._2
      case None => a => None
    }

    val hasSamples = vds.nSamples > 0

    val exportPP = options.exportPP

    def header: String = {
      val sb = new StringBuilder()

      sb.append("##fileformat=VCFv4.2\n")
      sb.append(s"##fileDate=${LocalDate.now}\n")
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
        readFile(options.append, state.hadoopConf) { s =>
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
      .map(_ => vds.queryVA("va.rsid")._2)

    val qualQuery: Option[Querier] = vas.getOption("qual")
      .map(_ => vds.queryVA("va.qual")._2)

    val filterQuery: Option[Querier] = vas.getOption("filters")
      .map(_ => vds.queryVA("va.filters")._2)

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
        sb.append(aa.alt))(() => sb += ',')
      sb += '\t'

      sb.append(qualQuery.flatMap(_ (a))
        .map(_.asInstanceOf[Double].formatted("%.2f"))
        .getOrElse("."))

      sb += '\t'

      filterQuery.flatMap(_ (a))
        .map(_.asInstanceOf[Set[String]]) match {
        case Some(f) =>
          if (f.nonEmpty)
            f.foreachBetween { s =>
              sb.append(s)
            } { () =>
              sb += ','
            }
          else
            sb += '.'
        case None => sb += '.'
      }

      sb += '\t'

      infoQuery(a).map(_.asInstanceOf[Row]) match {
        case Some(r) =>
          val toWrite =
            infoSignature.get.fields
              .zip(r.toSeq)
              .filter { case (f, v) => Option(v).isDefined }
          if (toWrite.isEmpty)
            sb += '.'
          else {
            toWrite
              .foreachBetween { case (f, v) =>
                sb.append(f.name)
                if (f.`type` != TBoolean) {
                  sb += '='
                  v match {
                    case i: Iterable[_] => i.foreachBetween { elem => sb.append(elem) } { () => sb.append(",") }
                    case _ => sb.append(v)
                  }
                }
              } { () => sb += ';' }
          }

        case None =>
          sb += '.'
      }

      if (hasSamples) {
        sb += '\t'
        if (exportPP)
          sb.append("GT:AD:DP:GQ:PP")
        else
          sb.append("GT:AD:DP:GQ:PL")
        gs.foreach { g =>
          sb += '\t'
          sb.append(g)
        }
      }
    }

    val kvRDD = vds.rdd.map { case (v, a, gs) =>
      (v, (a, gs.toGenotypeStream(v, compress = false)))
    }
    kvRDD.persist(StorageLevel.MEMORY_AND_DISK)
    kvRDD
      .repartitionAndSortWithinPartitions(new RangePartitioner[Variant, (Annotation, Iterable[Genotype])](vds.rdd.partitions.length, kvRDD))
      .mapPartitions { it: Iterator[(Variant, (Annotation, Iterable[Genotype]))] =>
        val sb = new StringBuilder
        it.map { case (v, (va, gs)) =>
          sb.clear()
          appendRow(sb, v, va, gs)
          sb.result()
        }
      }.writeTable(options.output, Some(header), deleteTmpFiles = true)
    kvRDD.unpersist()
    state
  }

}

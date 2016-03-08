package org.broadinstitute.hail.driver

import org.apache.spark.RangePartitioner
import org.apache.spark.storage.StorageLevel
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.variant.{Variant, Genotype}
import org.broadinstitute.hail.annotations.{VCFSignature, Annotations}
import org.kohsuke.args4j.{Option => Args4jOption}
import java.time._
import scala.io.Source

object ExportVCF extends Command {

  class Options extends BaseOptions {
    @Args4jOption(name = "-a", usage = "Append file to header")
    var append: String = _

    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _

  }

  def newOptions = new Options

  def name = "exportvcf"

  def description = "Write current dataset as VCF file"

  override def supportsMultiallelic = true

  def infoNumber(t: Type): String = t match {
    case TBoolean => "0"
    case TArray(elementType) => "."
    case _ => "1"
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

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val varAnnSig = vds.metadata.variantAnnotationSignatures

    def header: String = {
      val sb = new StringBuilder()

      sb.append("##fileformat=VCFv4.2\n")
      sb.append(s"##fileDate=${LocalDate.now}\n")
      // FIXME add Hail version
      sb.append(
        """##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
          |##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths for the ref and alt alleles in the order listed">
          |##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">
          |##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
          |##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Normalized, Phred-scaled likelihoods for genotypes as defined in the VCF specification">
          |""".stripMargin)

      vds.metadata.filters.map { case (key, desc) =>
        sb.append(s"""##FILTER=<ID=$key,Description="$desc">\n""")
      }

      vds.metadata.variantAnnotationSignatures
        .asInstanceOf[TStruct]
        .get("info")
        // FIXME
        .foreach(_.asInstanceOf[TStruct]
        .fields
        .foreach { case (_, f) =>
          sb.append("##INFO=<ID=")
          sb.append(f.name)
          sb.append(",Number=")
          sb.append(f.attr("Number").getOrElse(infoNumber(f.`type`)))
          sb.append(",Type=")
          sb.append(infoType(f.`type`))
          f.attr("Description") match {
            case Some(d) =>
              sb.append(",Description=\"")
              sb.append(d)
              sb += '"'
            case None =>
          }
          sb.append(">\n")
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

      sb.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")
      val sampleIds: Array[String] = vds.localSamples.map(vds.sampleIds)
      sampleIds.foreach { id =>
        sb += '\t'
        sb.append(id)
      }
      sb.result()
    }

    def printInfo(a: Any): String = {
      a match {
        case iter: Iterable[_] => iter.map(_.toString).mkString(",")
        case _ => a.toString
      }
    }

    def appendRow(sb: StringBuilder, v: Variant, a: Annotations, gs: Iterable[Genotype]) {
      sb.append(v.contig)
      sb += '\t'
      sb.append(v.start)
      sb += '\t'

      val id = a.getOption[String]("rsid")
        .getOrElse(".")
      sb.append(id)

      sb += '\t'
      sb.append(v.ref)
      sb += '\t'
      v.altAlleles.foreachBetween(aa =>
        sb.append(aa.alt))(() => sb += ',')
      sb += '\t'

      a.getOption[Double]("qual") match {
        case Some(d) => sb.append(d.formatted("%.2f"))
        case None => sb += '.'
      }
      sb += '\t'

      a.getOption[Set[String]]("filters") match {
        case Some(f) =>
          if (f.nonEmpty)
            f.foreachBetween(s => sb.append(s))(() => sb += ',')
          else
            sb += '.'
        case None => sb += '.'
      }

      sb += '\t'

      if (a.getOption[Annotations]("info").isDefined) {
        a.get[Annotations]("info").attrs
          .foreachBetween({ case (k, v) =>
            // FIXME handle cast error gracefully
            if (varAnnSig.asInstanceOf[TStruct].get("info").flatMap(_.asInstanceOf[TStruct].get(k)).get == TBoolean)
              sb.append(k)
            else {
              sb.append(k)
              sb += '='
              v match {
                case i: Iterable[_] => i.foreachBetween(elem => sb.append(elem))(() => sb.append(","))
                case _ => sb.append(v)
              }
            }
          })(() => sb += ';')
      } else
        sb += '.'

      sb += '\t'
      sb.append("GT:AD:DP:GQ:PL")

      gs.foreach { g =>
        sb += '\t'
        sb.append(g)
      }
    }

    val kvRDD = vds.rdd.map { case (v, a, gs) =>
      (v, (a, gs.toGenotypeStream(v, compress = false)))
    }
    kvRDD.persist(StorageLevel.MEMORY_AND_DISK)
    kvRDD
      .repartitionAndSortWithinPartitions(new RangePartitioner[Variant, (Annotations, Iterable[Genotype])](vds.rdd.partitions.length, kvRDD))
      .mapPartitions { it: Iterator[(Variant, (Annotations, Iterable[Genotype]))] =>
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

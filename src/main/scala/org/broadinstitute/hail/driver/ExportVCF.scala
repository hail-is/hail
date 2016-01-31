package org.broadinstitute.hail.driver

import org.apache.spark.RangePartitioner
import org.apache.spark.storage.StorageLevel
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.variant.{Variant, Genotype}
import org.broadinstitute.hail.annotations.{VCFSignature, Annotations}
import org.kohsuke.args4j.{Option => Args4jOption}
import java.time._

object ExportVCF extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _
  }

  def newOptions = new Options

  def name = "exportvcf"

  def description = "Write current dataset as VCF file"

  override def supportsMultiallelic = true

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

      val infoHeader = vds.metadata.variantAnnotationSignatures.getOption[Annotations]("info").map(_.attrs)
      infoHeader.foreach { i =>
        i.foreach { case (key, value) =>
          val sig = value.asInstanceOf[VCFSignature]
          sb.append(
            s"""##INFO=<ID=$key,Number=${sig.number},Type=${sig.vcfType},Description="${sig.description}">\n""")
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
        sb.append(aa.alt))(_ => sb += ',')
      sb += '\t'

      a.getOption[Double]("qual") match {
        case Some(d) => sb.append(d.formatted("%.2f"))
        case None => sb += '.'
      }
      sb += '\t'

      a.getOption[Set[String]]("filters") match {
        case Some(f) =>
          if (f.nonEmpty)
            f.foreachBetween(s => sb.append(s))(_ => sb += ',')
          else
            sb += '.'
        case None => sb += '.'
      }

      sb += '\t'

      if (a.getOption[Annotations]("info").isDefined) {
        a.get[Annotations]("info").attrs
          .foreachBetween({ case (k, v) =>
            if (varAnnSig.get[Annotations]("info").get[VCFSignature](k).vcfType == "Flag")
              sb.append(k)
            else {
              sb.append(k)
              sb += '='
              v match {
                case i: Iterable[_] => i.foreachBetween(elem => sb.append(elem))(_ => sb.append(","))
                case _ => sb.append(v)
              }
            }
          })(_ => sb += ';')
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

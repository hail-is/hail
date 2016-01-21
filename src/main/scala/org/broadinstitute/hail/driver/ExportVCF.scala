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

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val varAnnSig = vds.metadata.variantAnnotationSignatures

    def header: String = {
      val today = LocalDate.now.toString
      val sampleIds: Array[String] = vds.localSamples.map(vds.sampleIds)
      val version = "##fileformat=VCFv4.2\n"
      val date = s"##fileDate=$today\n"
      val source = "##source=Hailv0.0\n" // might be good to have a version variable

      val format =
        """##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths for the ref and alt alleles in the order listed">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Normalized, Phred-scaled likelihoods for genotypes as defined in the VCF specification">"""

      val filterHeader = vds.metadata.filters.map { case (key, desc) => s"""##FILTER=<ID=$key,Description="$desc">""" }.mkString("\n")

      val infoHeader = vds.metadata.variantAnnotationSignatures.attrs.get("info") match {
        case Some(anno: Annotations) => anno.attrs.map { case (key, sig) =>
          val vcfsig = sig.asInstanceOf[VCFSignature]
          val vcfsigType = vcfsig.vcfType
          val vcfsigNumber = vcfsig.number
          val vcfsigDesc = vcfsig.description
          s"""##INFO=<ID=$key,Number=$vcfsigNumber,Type=$vcfsigType,Description="$vcfsigDesc">"""
        }.mkString("\n") + "\n"
        case None => ""
        case _ => throw new UnsupportedOperationException("somebody put something bad in info")
      }

      val sb = new StringBuilder()
      sb.append(version)
      sb.append(date)
      sb.append(source)
      sb.append(format)
      sb.append("\n")
      sb.append(filterHeader)
      sb.append("\n")
      sb.append(infoHeader)

      val headerFragment = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT"
      sb.append(headerFragment)

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

      val id = a.attrs.get("rsid")
        .map(_.asInstanceOf[String])
        .getOrElse(".")
      sb.append(id)

      sb += '\t'
      sb.append(v.ref)
      sb += '\t'
      sb.append(v.alt)
      sb += '\t'

      val qual = a.attrs.get("qual")
        .map(_.asInstanceOf[Double])
        .map(_.formatted("%.2f"))
        .getOrElse(".")
      sb.append(qual)

      sb += '\t'

      val filter = a.attrs.get("filters")
        .map(_.asInstanceOf[Set[String]].mkString(","))
        .map(s => if (s.isEmpty) ".")
        .getOrElse(".")
      sb.append(filter)

      sb += '\t'

      var first = true
      val info = a.attrs.get("info")
        .map(_.asInstanceOf[Annotations].attrs
          .map {
            case (k, v) =>
              if (varAnnSig.attrs("info").asInstanceOf[Annotations].attrs(k).asInstanceOf[VCFSignature].vcfType == "Flag")
                k
              else
                s"$k=${printInfo(v)}"
          }
            .mkString(";")
        )
          .getOrElse(".")

      sb.append(info)

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

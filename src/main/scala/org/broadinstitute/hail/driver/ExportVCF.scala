package org.broadinstitute.hail.driver

import org.apache.spark.RangePartitioner
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.variant.{Variant,Genotype}
import org.broadinstitute.hail.annotations.AnnotationData
import org.kohsuke.args4j.{Option => Args4jOption}
import java.time._

object ExportVCF extends Command {
  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _
    @Args4jOption(required = true, name = "-t", aliases = Array("--tmpdir"), usage = "Directory for temporary files")
    var tmpdir: String = _
  }

  def newOptions = new Options

  def name = "exportvcf"

  def description = "Write current dataset as VCF file"

  def run(state:State,options:Options):State = {
    val vds = state.vds

    def header:String = {
      val today = LocalDate.now.toString
      val sampleIds:Array[String] = vds.localSamples.map(vds.sampleIds)
      val version = "##fileformat=VCFv4.2\n"
      val date = s"##fileDate=$today\n"
      val source = "##source=Hailv0.0\n" // might be good to have a version variable

      val format = """##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths for the ref and alt alleles in the order listed">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Normalized, Phred-scaled likelihoods for genotypes as defined in the VCF specification">"""

      val headerFragment = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"

      val sb = new StringBuilder()
      sb.append(version)
      sb.append(date)
      sb.append(source)
      sb.append(format)
      sb.append("\n")
      sb.append(headerFragment)
      sb.append(sampleIds.mkString("\t"))
      sb.append("\n")
      sb.result()
    }

    def vcfRow(v:Variant,a:AnnotationData,gs:Iterable[Genotype]):String = {
      val id = a.getVal("ID").getOrElse(".")
      val qual = a.getVal("QUAL").getOrElse(".")
      val filter = a.getVal("FILTER").getOrElse(".")
      val info = a.getVal("INFO").getOrElse(".")
      val format = "GT:AD:DP:GQ:PL"

      val sb = new StringBuilder()
      sb.append(v.contig)
      sb.append("\t")
      sb.append(v.start)
      sb.append("\t")
      sb.append(id)
      sb.append("\t")
      sb.append(v.ref)
      sb.append("\t")
      sb.append(v.alt)
      sb.append("\t")
      sb.append(qual)
      sb.append("\t")
      sb.append(filter)
      sb.append("\t")
      sb.append(info)
      sb.append("\t")
      sb.append(format)
      sb.append("\t")
      sb.append(gs.map{_.toString}.mkString("\t"))
      sb.result()
    }

    hadoopDelete(options.output, state.hadoopConf, true)

    vds.rdd
      .map{case (v,a,gs) => (v,(a,gs))}
      .repartitionAndSortWithinPartitions(new RangePartitioner[Variant,(AnnotationData,Iterable[Genotype])](vds.rdd.partitions.length,vds.rdd.map{case (v,a,gs) => (v,(a,gs))}))
      .map{case (v,(a,gs)) => vcfRow(v,a,gs)}
      .writeTableSingleFile(options.output,header,options.tmpdir,true,true)
    state
  }
}

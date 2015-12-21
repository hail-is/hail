package org.broadinstitute.hail.driver

import org.apache.spark.RangePartitioner
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.variant.{Variant,Genotype}
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

      val format = Array("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
          "##FORMAT=<ID=AD,Number=R,Type=Integer,Description=\"Allelic depths for the ref and alt alleles in the order listed\">",
          "##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Read Depth\">",
          "##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Genotype Quality\">",
          "##FORMAT=<ID=PL,Number=G,Type=Integer,Description=\"Normalized, Phred-scaled likelihoods for genotypes as defined in the VCF specification\">").mkString("\n") + "\n"

      val headerrow = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + sampleIds.mkString("\t") + "\n"

      version + date + source + format + headerrow
    }

    def vcfRow(v:Variant,gs:Iterable[Genotype]):String = {
      val id = "." //get this from tim's annotations
      val qual = "." //get this from tim's annotations
      val filter = "." //get this from tim's annotations
      val info = "." //get this from tim's annotations
      val format = "GT:AD:DP:GQ:PL"

      val metadata = Array(v.contig, v.start,id, v.ref, v.alt, qual, filter, info, format)
      val data = gs.map{_.toString}.mkString("\t")

      metadata.mkString("\t") + "\t" + data
    }

    hadoopDelete(options.output, state.hadoopConf, true)

    vds.rdd.repartitionAndSortWithinPartitions(new RangePartitioner[Variant,Iterable[Genotype]](vds.rdd.partitions.length,vds.rdd))
      .map{case (v,gs) => vcfRow(v,gs)}.writeSingleFile(options.output,header,options.tmpdir)
    state
  }
}

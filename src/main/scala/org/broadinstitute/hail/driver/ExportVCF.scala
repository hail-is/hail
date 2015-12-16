package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.variant.{VariantDataset,Variant,Genotype}
import org.kohsuke.args4j.{Option => Args4jOption}

object ExportVCF extends Command {
  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _
  }

  def newOptions = new Options

  def name = "exportvcf"

  def description = "Write current dataset as VCF file"

  def run(state:State,options:Options):State = {
    val vds = state.vds
    hadoopDelete(options.output, state.hadoopConf, true)

    def header:String = {
      val sampleIds:Array[String] = vds.localSamples.map(vds.sampleIds)
      val version = "##fileformat=VCFv4.2\n"
      val date = "##fileDate=20151215\n"
      val source = "##source=HailV0.0\n" // might be good to have a version variable
      val header = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + sampleIds.mkString("\t") + "\n"
      version + date + source + header
    }

    def vcfRow(v:Variant,gs:Iterable[Genotype]):String = {
      val id = "." //get this from tim
      val qual = "." //get this from tim
      val filter = "." //get this from tim
      val info = "." //get this from tim
      val format = "GT:AD:DP:GQ:PL"

      val metadata = Array(v.contig, v.start,id, v.ref, v.alt, qual, filter, info, format)
      val data = gs.map{_.toString}.mkString("\t")

      metadata.mkString("\t") + "\t" + data
    }

    vds.rdd.map{case (v,gs) => vcfRow(v,gs)}.writeFlatFile(options.output,header)
    state
  }
}

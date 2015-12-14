package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.variant.{VariantDataset,Variant,Genotype}
import org.kohsuke.args4j.{Option => Args4jOption}

object Write extends Command {
  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _

    @Args4jOption(required = true, name = "-t", aliases = Array("--type"), usage = "Output file type")
    var outputType: String = _
  }

  def newOptions = new Options

  def name = "write"


  def description = newOptions.outputType match {
    case "vds" => "Write current dataset as .vds file"
    case "vcf" => "Write current dataset as VCF file"
  }

  def run(state: State, options: Options): State = {
    options.outputType match {
      case "vds" => writeVds(state,options)
      case "vcf" => writeVcf(state,options)
    }
  }

  def writeVds(state:State,options:Options):State = {
    hadoopDelete(options.output, state.hadoopConf, true)
    state.vds.write(state.sqlContext, options.output)
    state
  }

  def writeVcf(state:State,options:Options):State = {
    hadoopDelete(options.output, state.hadoopConf, true)

    def header(vds:VariantDataset):String = {
      val sampleIds:Array[String] = vds.localSamples.map(vds.sampleIds)
      val header = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + sampleIds.mkString("\t")
      header
    }

    def vcfRow(v:Variant,gs:Iterable[Genotype]):String = {
      val id = "NA"
      val qual = "NA"
      val filter = "NA"
      val info = "NA"
      val format = "NA"

      val metadata = Array(v.contig, v.start,id, v.ref, v.alt, qual, filter, info, format)
      val data = gs.map{_.toString}.mkString("\t")

      metadata.mkString("\t") + "\t" + data
    }

    state.vds.rdd.map{case (v,gs) => vcfRow(v,gs)}.writeFlatFile(options.output,header(state.vds))
    state
    //write to hadoop (compressed if needed)
    //merge hadoop files

  }
}

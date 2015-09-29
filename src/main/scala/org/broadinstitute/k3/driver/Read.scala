package org.broadinstitute.k3.driver

import org.broadinstitute.k3.methods.LoadVCF
import org.broadinstitute.k3.variant.{Genotype, VariantSampleMatrix}
import org.broadinstitute.k3.Utils._
import org.kohsuke.args4j.{Option => Args4jOption}

object Read extends Command {
  def name = "read"
  def description = "Load file (.vds, .vcf or .vcf.bgz) as the current dataset"

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-i", aliases = Array("--input"), usage = "Input file")
    var input: String = _

    // FIXME not needed: use read vs readvcf?
    @Args4jOption(required = false, name = "-m", aliases = Array("--vsm-type"), usage = "Select VariantSampleMatrix implementation")
    var vsmtype: String = "sparky"

    @Args4jOption(required = false, name = "-n", aliases = Array("--npartition"), usage = "Number of partitions")
    var nPartitions: Int = 0
  }
  def newOptions = new Options

  def run(state: State, options: Options): State = {
    val input = options.input
    val newVDS =
      if (input.endsWith(".vds"))
        VariantSampleMatrix.read(state.sqlContext, input)
      else if (input.endsWith(".vcf")
        || input.endsWith(".vcf.bgz")
        || input.endsWith(".vcf.gz"))
        LoadVCF(state.sc, options.vsmtype, input,
          if (options.nPartitions != 0)
            Some(options.nPartitions)
          else
            None)
      else
        fatal("unknown input file type")

    state.copy(vds = newVDS)
  }
}

package org.broadinstitute.hail.driver

import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.variant.VariantSampleMatrix
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.vcf
import org.kohsuke.args4j.{Option => Args4jOption}

object Import extends Command {
  def name = "import"

  def description = "Load file (.vcf or .vcf.bgz) as the current dataset"

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-i", aliases = Array("--input"), usage = "Input file")
    var input: String = _

    @Args4jOption(required = false, name = "-m", aliases = Array("--vsm-type"), usage = "Select VariantSampleMatrix implementation")
    var vsmtype: String = "sparky"

    @Args4jOption(required = false, name = "-d", aliases = Array("--no-compress"), usage = "Don't compress in-memory representation")
    var noCompress: Boolean = false

    @Args4jOption(required = false, name = "-n", aliases = Array("--npartition"), usage = "Number of partitions")
    var nPartitions: Int = 0

    @Args4jOption(required = false, name = "-f", aliases = Array("--force"), usage = "Force load .gz file")
    var force: Boolean = false
  }

  def newOptions = new Options

  def run(state: State, options: Options): State = {
    val input = options.input

    val newVDS =
      if (input.endsWith(".vcf")
        || input.endsWith(".vcf.bgz")
        || input.endsWith(".vcf.gz")) {
        if (!options.force
          && input.endsWith(".gz")) {
          fatal(".gz cannot be loaded in parallel, use .bgz or -f override")
        }

        LoadVCF(state.sc, input, options.vsmtype, !options.noCompress,
          if (options.nPartitions != 0)
            Some(options.nPartitions)
          else
            None)
      } else
        fatal("unknown input file type")

    state.copy(vds = newVDS)
  }
}

package org.broadinstitute.hail.driver

import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.io.annotators.IntervalListAnnotator
import org.kohsuke.args4j.{Option => Args4jOption}

object FilterVariantsIntervals extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "-i", aliases = Array("--input"),
      usage = "Path to interval list file")
    var input: String = _

    @Args4jOption(required = false, name = "--keep", usage = "Keep variants matching condition")
    var keep: Boolean = false

    @Args4jOption(required = false, name = "--remove", usage = "Remove variants matching condition")
    var remove: Boolean = false

  }

  def newOptions = new Options

  def name = "filtervariants intervals"

  def description = "Filter variants in current dataset with an interval list"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    if (!(options.keep ^ options.remove))
      fatal("either `--keep' or `--remove' required, but not both")

    val cond = options.input
    val keep = options.keep
    val ilist = IntervalListAnnotator.read(options.input, state.hadoopConf, prune = true)

    state.copy(vds = vds.filterIntervals(ilist, options.keep))
  }
}

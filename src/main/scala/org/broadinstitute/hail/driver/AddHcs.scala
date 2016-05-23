package org.broadinstitute.hail.driver

import org.broadinstitute.hail.variant.HardCallSet
import org.kohsuke.args4j.{Option => Args4jOption}

object AddHcs extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "-s", aliases = Array("--sparse"), usage = "Sparse cut off, < s is sparse: s <= 0 is all dense, s > 1 is all sparse")
    var sparseCutOff: Double = .15

    @Args4jOption(required = false, name = "-b", aliases = Array("--blockwidth"), usage = "Width of DataFrame block in basepairs")
    var blockWidth: Int = 100000
  }

  def supportsMultiallelic = false

  def requiresVDS = true

  def newOptions = new Options

  def name = "addhcs"

  def description = "Add hard call set to state"

  def run(state: State, options: Options): State = {
    state.copy(hcs =
      HardCallSet(
        state.sqlContext,
        state.vds,
        sparseCutoff = options.sparseCutOff,
        blockWidth = options.blockWidth))
  }
}
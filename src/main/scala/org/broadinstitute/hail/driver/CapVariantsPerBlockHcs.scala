package org.broadinstitute.hail.driver

import org.broadinstitute.hail.utils._
import org.kohsuke.args4j.{Option => Args4jOption}

object CapVariantsPerBlockHcs extends Command {
  def name = "caphcs"

  def description = "Cap the number of variants per current block and then optionally update the block size"

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-c", aliases = Array("--cap"), usage = "Maximum number of variants per current block")
    var cap: Int = _

    @Args4jOption(required = false, name = "-b", aliases = Array("--blockwidth"), usage = "New width of block in basepairs")
    var blockWidth: Int = _
  }

  def newOptions = new Options

  def run(state: State, options: Options): State = {
    if (state.hcs == null)
      fatal("Run addhcs before caphcs to add hard call set to state")

    val newHcs = state.hcs.capNVariantsPerBlock(options.cap, options.blockWidth)

    state.copy(hcs = newHcs)
  }

  def supportsMultiallelic = true

  def requiresVDS = false
}

package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.kohsuke.args4j.{Option => Args4jOption}

object CapVariantsPerBlockHcs extends Command {
  def name = "caphcs"

  def description = "Load .hcs file into state"

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-c", aliases = Array("--cap"), usage = "Maximum number of variants per block")
    var cap: Int = _
  }

  def newOptions = new Options

  def run(state: State, options: Options): State = {
    if (state.hcs == null)
      fatal("Run addhcs before caphcs to add hard call set to state")

    val newHcs = state.hcs.capVariantsPerBlock(options.cap)

    state.copy(hcs = newHcs)
  }
}

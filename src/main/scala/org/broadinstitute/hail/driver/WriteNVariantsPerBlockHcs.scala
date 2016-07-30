package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.kohsuke.args4j.{Option => Args4jOption}

object WriteNVariantsPerBlockHcs extends Command {
  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _
  }

  def newOptions = new Options

  def name = "writenvariantsperblock"
  def description = "Write count of variants per block of current hard call set as sorted .tsv file"

  def run(state: State, options: Options): State = {
    if (state.hcs == null)
      fatal("Run addhcs before writehcs to add hard call set to state")

    state.hcs.writeNVariantsPerBlock(options.output)

    state
  }

  def supportsMultiallelic = false

  def requiresVDS = false
}

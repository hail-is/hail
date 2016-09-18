package org.broadinstitute.hail.driver

import org.broadinstitute.hail.utils._
import org.kohsuke.args4j.{Option => Args4jOption}

object WriteHcs extends Command {
  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _
  }

  def newOptions = new Options

  def name = "writehcs"
  def description = "Write current hard call set as .hcs file"

  def run(state: State, options: Options): State = {
    if (state.hcs == null)
      fatal("Run addhcs before writehcs to add hard call set to state")

    state.hadoopConf.delete(options.output, recursive = true) // overwrite by default
    state.hcs.write(state.sqlContext, options.output)

    state
  }

  def supportsMultiallelic = true

  def requiresVDS = false
}

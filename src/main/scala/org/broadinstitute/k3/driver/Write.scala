package org.broadinstitute.k3.driver

import org.kohsuke.args4j.{Option => Args4jOption}

object Write extends Command {
  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _
  }
  def newOptions = new Options

  def name = "write"
  def description = "Write current dataset as .vds file"

  def run(state: State, options: Options): State = {
    state.vds.write(state.sqlContext, options.output)
    state
  }
}

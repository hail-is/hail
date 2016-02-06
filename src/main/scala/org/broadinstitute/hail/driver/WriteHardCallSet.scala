package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.variant.HardCallSet
import org.kohsuke.args4j.{Option => Args4jOption}

object WriteHardCallSet extends Command {
  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _

  }
  def newOptions = new Options

  def name = "writehcs"
  def description = "Write current dataset as .hcs file"

  def run(state: State, options: Options): State = {
    hadoopDelete(options.output, state.hadoopConf, true)
    HardCallSet(state.vds).write(state.sqlContext, options.output)
    state
  }
}

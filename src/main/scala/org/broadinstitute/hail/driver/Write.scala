package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.kohsuke.args4j.{Option => Args4jOption}

object Write extends Command {
  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _

    @Args4jOption(required = false, name = "--no-compress", usage = "Don't compress genotype streams")
    var noCompress: Boolean = false
  }
  def newOptions = new Options

  def name = "write"

  def description = "Write current dataset as .vds file"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    hadoopDelete(options.output, state.hadoopConf, true)
    state.vds.write(state.sqlContext, options.output, compress = !options.noCompress)
    state
  }
}

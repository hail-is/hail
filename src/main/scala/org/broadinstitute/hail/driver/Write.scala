package org.broadinstitute.hail.driver

import org.broadinstitute.hail.utils._
import org.kohsuke.args4j.{Option => Args4jOption}

object Write extends Command {
  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _

    @Args4jOption(required = false, name = "--no-compress", usage = "Don't compress genotype streams")
    var noCompress: Boolean = false

    @Args4jOption(required = false, name = "--overwrite", usage = "Delete existing file at output location")
    var overwrite: Boolean = false
  }
  def newOptions = new Options

  def name = "write"

  def description = "Write current dataset as .vds file"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    if (options.overwrite)
      state.hadoopConf.delete(options.output, recursive =  true)
    else if (state.hadoopConf.exists(options.output))
      fatal(
        s"""File already exists at ${options.output}
           |  Choose this path or rerun with --overwrite flag,
           |  but do NOT overwrite a VDS being read.""".stripMargin)
    state.vds.write(state.sqlContext, options.output, compress = !options.noCompress)
    state
  }
}

package org.broadinstitute.hail.driver

import org.broadinstitute.hail.methods.Pedigree
import org.kohsuke.args4j.{Option => Args4jOption}

object FamSummary extends Command {
  def name = "famsummary"
  def description = "Summarize a .fam file"

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-f", aliases = Array("--fam"), usage = ".fam file")
    var famFilename: String = _

    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _
  }
  def newOptions = new Options

  def supportsMultiallelic = true

  def requiresVDS = false

  def run(state: State, options: Options): State = {
    val ped = Pedigree.read(options.famFilename, state.hadoopConf, state.vds.sampleIds)
    ped.writeSummary(options.output, state.hadoopConf)

    state
  }
}

package org.broadinstitute.hail.driver

import org.kohsuke.args4j.{Option => Args4jOption}

object Head extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "--keep", usage = "Number (N) of variants to keep")
    var keep: Int = _
  }

  def newOptions = new Options

  def name = "head"

  def description = "Takes N variants in current dataset from the first partition(s) (NOT IN GENOMIC ORDER)"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds
    state.copy(vds = vds.head(options.keep))
  }
}

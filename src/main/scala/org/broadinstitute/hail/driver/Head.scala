package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.kohsuke.args4j.{Option => Args4jOption}

object Head extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "--keep", usage = "(Expected) number of variants to keep")
    var keep: Long = _
  }

  def newOptions = new Options

  def name = "head"

  def description = "Takes ~N variants in current dataset from the first partition(s) (NOT IN GENOMIC ORDER)"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds
    state.copy(vds = vds.head(options.keep))
  }
}

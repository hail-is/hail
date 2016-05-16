package org.broadinstitute.hail.driver

import org.kohsuke.args4j.{Option => Args4jOption}

object Repartition extends Command {
  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-n", aliases = Array("--partitions"), usage = "Number of partitions")
    var k: Int = _
  }
  def newOptions = new Options

  def supportsMultiallelic = true

  def requiresVDS = true

  def name = "repartition"
  def description = "Repartition the current dataset"

  def run(state: State, options: Options): State = {
    state.copy(vds = state.vds.repartition(options.k))
  }
}

package org.broadinstitute.hail.driver

import org.kohsuke.args4j.{Option => Args4jOption}
import org.broadinstitute.hail.Utils._

object RepartitionHcs extends Command {
  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-n", aliases = Array("--partitions"), usage = "Number of partitions")
    var n: Integer = _
  }
  def newOptions = new Options

  def name = "repartitionhcs"
  def description = "Repartition the current hard call set"

  def run(state: State, options: Options): State = {

    if (options.n < 1)
        fatal("Number of partitions must be positive.")

    state.copy(hcs = state.hcs.repartition(options.n))
  }

  def supportsMultiallelic = true

  def requiresVDS = false
}

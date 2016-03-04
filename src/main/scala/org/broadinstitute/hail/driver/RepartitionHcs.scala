package org.broadinstitute.hail.driver

import org.kohsuke.args4j.{Option => Args4jOption}

object RepartitionHcs extends Command {
  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-n", aliases = Array("--partitions"), usage = "Number of partitions")
    var k: Int = _
  }
  def newOptions = new Options

  def name = "repartitionhcs"
  def description = "Repartition the current hard call set"

  def run(state: State, options: Options): State = {
    state.copy(hcs = state.hcs.repartition(options.k))
  }
}

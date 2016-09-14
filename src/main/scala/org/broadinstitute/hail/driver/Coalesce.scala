package org.broadinstitute.hail.driver

import org.kohsuke.args4j.{Option => Args4jOption}
import org.broadinstitute.hail.utils._

object Coalesce extends Command {

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
    val n = state.vds.nPartitions
    val k = options.k
    if (k < 1)
      fatal(
        s"""invalid `partitions' argument: $k
            |  Must request positive number of partitions""".stripMargin)
    else if (n < k) {
      warn(
        s"""cannot coalesce to a larger number of partitions:
            |  Dataset has $n partitions, requested $k partitions
            |  In order to run with more partitions, use the --blocksize global option and reimport.""".stripMargin)
    }

    state.copy(vds = state.vds.coalesce(k))
  }
}

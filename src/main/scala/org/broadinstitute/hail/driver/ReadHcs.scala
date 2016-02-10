package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.variant.HardCallSet
import org.kohsuke.args4j.{Option => Args4jOption}

object ReadHcs extends Command {
  def name = "readhcs"

  def description = "Load .hcs file into state"

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-i", aliases = Array("--input"), usage = "Input file")
    var input: String = _

    @Args4jOption(required = false, name = "-n", aliases = Array("--npartition"), usage = "Number of partitions")
    var nPartitions: Int = 0
  }

  def newOptions = new Options

  def run(state: State, options: Options): State = {
    val input = options.input

    val newHcs =
      if (input.endsWith(".hcs"))
        HardCallSet.read(state.sqlContext, options.input)
      else
        fatal("unknown input file type")

    state.copy(hcs = newHcs)
  }
}

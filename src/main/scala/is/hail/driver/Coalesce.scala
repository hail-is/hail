package is.hail.driver

import org.kohsuke.args4j.{Option => Args4jOption}
import is.hail.utils._

object Coalesce extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-n", aliases = Array("--partitions"), usage = "Number of partitions")
    var k: Int = _

    @Args4jOption(required = false, name = "--no-shuffle", usage = "don't shuffle in repartition")
    var noShuffle: Boolean = false
  }

  def newOptions = new Options

  def supportsMultiallelic = true

  def requiresVDS = true

  def name = "repartition"

  def description = "Increase or decrease the dataset sharding.  Can improve performance after large filters."

  def run(state: State, options: Options): State = {
    val n = state.vds.nPartitions
    val k = options.k
    val shuffle = !options.noShuffle
    if (k < 1)
      fatal(
        s"""invalid `partitions' argument: $k
            |  Must request positive number of partitions""".stripMargin)
    else if (n == k)
      warn(s"""dataset already had exactly $k partitions, repartition had no effect""")
    else if ((n < k) && !shuffle)
      warn(
        s"""cannot coalesce to a larger number of partitions with --no-shuffle option:
            |  Dataset has $n partitions, requested $k partitions""".stripMargin)

    state.copy(vds = state.vds
      .coalesce(k, shuffle = shuffle))
  }
}

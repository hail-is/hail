package is.hail.driver

import org.apache.spark.rdd.RDD
import is.hail.utils._
import is.hail.variant._
import org.kohsuke.args4j.{Option => Args4jOption}

object FilterVariantsList extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "-i", aliases = Array("--input"),
      usage = "Path to variant list file")
    var input: String = _

    @Args4jOption(required = false, name = "--keep", usage = "Keep variants matching condition")
    var keep: Boolean = false

    @Args4jOption(required = false, name = "--remove", usage = "Remove variants matching condition")
    var remove: Boolean = false
  }

  def newOptions = new Options

  def name = "filtervariants list"

  def description = "Filter variants in current dataset with a variant list"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    if (!(options.keep ^ options.remove))
      fatal("either `--keep' or `--remove' required, but not both")

    state.copy(vds = vds.filterVariantsList(options.input, options.keep))
  }
}

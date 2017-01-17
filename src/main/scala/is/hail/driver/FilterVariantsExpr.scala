package is.hail.driver

import is.hail.utils._
import is.hail.annotations._
import is.hail.expr._
import is.hail.methods._
import is.hail.variant._
import org.kohsuke.args4j.{Option => Args4jOption}

object FilterVariantsExpr extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "-c", aliases = Array("--condition"),
      usage = "Filter expression involving `v' (variant) and `va' (variant annotations)")
    var condition: String = _

    @Args4jOption(required = false, name = "--keep", usage = "Keep variants matching condition")
    var keep: Boolean = false

    @Args4jOption(required = false, name = "--remove", usage = "Remove variants matching condition")
    var remove: Boolean = false

  }

  def newOptions = new Options

  def name = "filtervariants expr"

  def description = "Filter variants in current dataset using the Hail expression language"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    if (!(options.keep ^ options.remove))
      fatal("either `--keep' or `--remove' required, but not both")

    val cond = options.condition
    val keep = options.keep

    state.copy(vds = vds.filterVariantsExpr(cond, keep))
  }
}

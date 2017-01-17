package is.hail.driver

import is.hail.methods.CalculateConcordance
import is.hail.utils._
import is.hail.variant._
import org.kohsuke.args4j.{Option => Args4jOption}

object Concordance extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-r", aliases = Array("--right"),
      usage = "Name of dataset in environment to join on the right")
    var rightName: String = _

    @Args4jOption(required = true, name = "-v", aliases = Array("--variants"),
      usage = "Environment name to give resulting sites-only VDS")
    var variantName: String = _

    @Args4jOption(required = true, name = "-s", aliases = Array("--samples"),
      usage = "Environment name to give resulting samples-only VDS")
    var sampleName: String = _
  }

  def newOptions = new Options

  def name = "concordance"

  def description = "Calculate call concordance between two datasets.  Does an inner join on samples, outer join on variants."

  def supportsMultiallelic = false

  def requiresVDS = true

  def calculate(left: VariantDataset, right: VariantDataset): (VariantDataset, VariantDataset) = {
    if (!right.wasSplit) {
      fatal(
        s"""The right dataset was not split
            |  Run `splitmulti' on this dataset before calculating concordance.""".stripMargin)
    }

    val (samples, variants) = CalculateConcordance(left, right)

    (samples, variants)
  }

  def run(state: State, options: Options): State = {
    val right = state.env.getOrElse(options.rightName, fatal(s"no such dataset $name in environment"))
    val (samples, variants) = calculate(state.vds, right)

    info(s"Storing sites-only VDS with global and variant concordance annotations in environment as `${ options.variantName }'")
    info(s"Storing samples-only VDS with global and sample concordance annotations in environment as `${ options.sampleName }'")

    state.copy(env = state.env + (options.sampleName -> samples) + (options.variantName -> variants))
  }
}

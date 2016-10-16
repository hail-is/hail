package org.broadinstitute.hail.driver

import org.broadinstitute.hail.methods.CalculateConcordance
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant._
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

  def run(state: State, options: Options): State = {
    val left = state.vds
    val rightName = options.rightName

    val right = state.env.getOrElse(rightName, fatal(s"no such dataset $name in environment"))

    if (!right.wasSplit) {
      fatal(
        s"""The right dataset $rightName was not split
            |  Run `splitmulti' on this dataset before calculating concordance.
         """.stripMargin)
    }

    val (samples, variants) = CalculateConcordance(left, right)

    state.copy(env = state.env + (options.sampleName -> samples) + (options.variantName -> variants))
  }
}

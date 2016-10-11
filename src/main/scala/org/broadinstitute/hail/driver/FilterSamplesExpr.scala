package org.broadinstitute.hail.driver

import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods._
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable.ArrayBuffer

object FilterSamplesExpr extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "-c", aliases = Array("--condition"),
      usage = "Filter expression involving `s' (sample) and `sa' (sample annotations)")
    var condition: String = _

    @Args4jOption(required = false, name = "--keep", usage = "Keep only listed samples in current dataset")
    var keep: Boolean = false

    @Args4jOption(required = false, name = "--remove", usage = "Remove listed samples from current dataset")
    var remove: Boolean = false
  }

  def newOptions = new Options

  def name = "filtersamples expr"

  def description = "Filter samples in current dataset using the Hail expression language"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    if (!(options.keep ^ options.remove))
      fatal("either `--keep' or `--remove' required, but not both")

    val keep = options.keep

    state.copy(vds = state.vds.filterSamples(options.condition, keep))
  }
}

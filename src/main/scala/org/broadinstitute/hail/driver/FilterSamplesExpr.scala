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
    val localGlobalAnnotation = vds.globalAnnotation

    if (!(options.keep ^ options.remove))
      fatal("either `--keep' or `--remove' required, but not both")

    val keep = options.keep
    val sas = vds.saSignature
    val cond = options.condition

    val ec = Aggregators.sampleEC(vds)

    val f: () => Option[Boolean] = Parser.parseTypedExpr[Boolean](cond, ec)

    val sampleAggregationOption = Aggregators.buildSampleAggregations(vds, ec)

    val sampleIds = state.vds.sampleIds
    val p = (s: String, sa: Annotation) => {
      sampleAggregationOption.foreach(f => f.apply(s))
      ec.setAll(localGlobalAnnotation, s, sa)
      Filter.keepThis(f(), keep)
    }

    state.copy(vds = vds.filterSamples(p))
  }
}

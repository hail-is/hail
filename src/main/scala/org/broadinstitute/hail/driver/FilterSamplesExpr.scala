package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
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

    if ((options.keep && options.remove)
      || (!options.keep && !options.remove))
      fatal("one `--keep' or `--remove' required, but not both")

    val keep = options.keep
    val sas = vds.saSignature
    val cond = options.condition
    val aggregationEC = EvalContext(Map(
      "v" ->(0, TVariant),
      "va" ->(1, vds.vaSignature),
      "s" ->(2, TSample),
      "sa" ->(3, vds.saSignature),
      "g" ->(4, TGenotype),
      "global" ->(5, vds.globalSignature)))

    val symTab = Map(
      "s" ->(0, TSample),
      "sa" ->(1, vds.saSignature),
      "global" ->(2, vds.globalSignature),
      "gs" ->(-1, TAggregable(aggregationEC)))

    val ec = EvalContext(symTab)
    ec.set(2, vds.globalAnnotation)
    aggregationEC.set(5, vds.globalAnnotation)
    val f: () => Option[Boolean] = Parser.parse[Boolean](cond, ec, TBoolean)

    val aggregatorA = aggregationEC.a
    val aggregators = ec.aggregationFunctions

    val sampleAggregationOption = Aggregators.buildSampleAggregations(vds, aggregationEC)


    val sampleIds = state.vds.sampleIds
    val p = (s: String, sa: Annotation) => {
      ec.setAll(s, sa)

      sampleAggregationOption.foreach(f => f.apply(s))

      Filter.keepThis(f(), keep)
    }

    state.copy(vds = vds.filterSamples(p))
  }
}

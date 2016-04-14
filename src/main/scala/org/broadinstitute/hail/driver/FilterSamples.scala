package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.variant.{Sample, VariantDataset}
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

object FilterSamples extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "--keep", usage = "Keep only listed samples in current dataset")
    var keep: Boolean = false

    @Args4jOption(required = false, name = "--remove", usage = "Remove listed samples from current dataset")
    var remove: Boolean = false

    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = "Filter condition: expression or .sample_list file (one sample name per line)")
    var condition: String = _
  }

  def newOptions = new Options

  def name = "filtersamples"

  def description = "Filter samples in current dataset"

  override def supportsMultiallelic = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    if (!options.keep && !options.remove)
      fatal(name + ": one of `--keep' or `--remove' required")

    val keep = options.keep
    val sas = vds.saSignature

    val cond = options.condition
    val p = cond match {
      case f if f.endsWith(".sample_list") =>
        val indexOfSample: Map[String, Int] = vds.sampleIds.zipWithIndex.toMap
        val samples = readFile(f, state.hadoopConf) { reader =>
          Source.fromInputStream(reader)
            .getLines()
            .filter(line => !line.isEmpty)
            .toSet
        }
        (s: String, sa: Annotation) => Filter.keepThis(samples.contains(s), keep)
      case c: String =>
        val aggregationEC = EvalContext(Map(
          "v" ->(0, TVariant),
          "va" ->(1, vds.vaSignature),
          "s" ->(2, TSample),
          "sa" ->(3, sas),
          "g" ->(4, TGenotype)))
        val symTab = Map(
          "s" ->(0, TSample),
          "sa" ->(1, sas),
        "gs" -> (-1, TAggregable(aggregationEC)))
        val ec = EvalContext(symTab)
        val f: () => Option[Boolean] = Parser.parse[Boolean](cond, ec, TBoolean)


        val a = ec.a
        val aggregatorA = aggregationEC.a
        val aggregators = ec.aggregationFunctions

        val doAggregates = aggregators.nonEmpty

        val sampleAggregationOption = Aggregators.buildSampleAggregations(vds, aggregationEC)


        val sampleIds = state.vds.sampleIds
        (s: String, sa: Annotation) => {
          a(0) = s
          a(1) = sa

          sampleAggregationOption.foreach(f => f.apply(s))

          Filter.keepThis(f(), keep)
        }
    }

    state.copy(vds = vds.filterSamples(p))
  }
}

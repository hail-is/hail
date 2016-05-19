package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods._
import org.kohsuke.args4j.{Option => Args4jOption}

object ExportSamples extends Command {

  class Options extends BaseOptions {

    @Args4jOption(required = true, name = "-o", aliases = Array("--output"),
      usage = "path of output tsv")
    var output: String = _

    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = ".columns file, or comma-separated list of fields/computations to be printed to tsv")
    var condition: String = _
  }

  def newOptions = new Options

  def name = "exportsamples"

  def description = "Export list of sample information to tsv"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val hConf = vds.sparkContext.hadoopConfiguration
    val sas = vds.saSignature
    val cond = options.condition
    val output = options.output

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

    val (header, fs) = if (cond.endsWith(".columns"))
      ExportTSV.parseColumnsFile(ec, cond, vds.sparkContext.hadoopConfiguration)
    else
      Parser.parseExportArgs(cond, ec)

    val aggregatorA = aggregationEC.a

    val sampleAggregationOption = Aggregators.buildSampleAggregations(vds, aggregationEC)

    hadoopDelete(output, state.hadoopConf, recursive = true)

    val sb = new StringBuilder()
    val lines = for ((s, sa) <- vds.sampleIdsAndAnnotations) yield {
      sb.clear()

      ec.setAll(s, sa)

      sampleAggregationOption.foreach(f => f.apply(s))

      var first = true
      fs.iterator.foreachBetween(f => sb.tsvAppend(f()))(() => sb += '\t')
      sb.result()
    }

    writeTable(output, hConf, lines, header)

    state
  }

}

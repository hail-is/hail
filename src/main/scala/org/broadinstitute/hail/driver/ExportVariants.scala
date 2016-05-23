package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods._
import org.kohsuke.args4j.{Option => Args4jOption}

object ExportVariants extends Command {

  class Options extends BaseOptions {

    @Args4jOption(required = true, name = "-o", aliases = Array("--output"),
      usage = "path of output tsv")
    var output: String = _

    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = ".columns file, or comma-separated list of fields/computations to be printed to tsv")
    var condition: String = _
  }

  def newOptions = new Options

  def name = "exportvariants"

  def description = "Export list of variant information to tsv"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val vas = vds.vaSignature
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
      "v" ->(0, TVariant),
      "va" ->(1, vds.vaSignature),
      "global" ->(2, vds.globalSignature),
      "gs" ->(-1, TAggregable(aggregationEC)))


    val ec = EvalContext(symTab)
    ec.set(2, vds.globalAnnotation)
    aggregationEC.set(5, vds.globalAnnotation)

    val (header, fs) = if (cond.endsWith(".columns"))
      ExportTSV.parseColumnsFile(ec, cond, vds.sparkContext.hadoopConfiguration)
    else
      Parser.parseExportArgs(cond, ec)

    val variantAggregations = Aggregators.buildVariantaggregations(vds, aggregationEC)

    hadoopDelete(output, state.hadoopConf, recursive = true)

    vds.rdd
      .mapPartitions { it =>
        val sb = new StringBuilder()
        it.map { case (v, va, gs) =>

          variantAggregations.foreach { f => f(v, va, gs)}
          sb.clear()

          ec.setAll(v, va)

          fs.iterator.foreachBetween { f => sb.tsvAppend(f()) }(() => sb.append("\t"))
          sb.result()
        }
      }.writeTable(output, header)

    state
  }
}

package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods._
import org.kohsuke.args4j.{Option => Args4jOption}
import scala.collection.mutable.ArrayBuffer

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

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val hConf = vds.sparkContext.hadoopConfiguration
    val sas = vds.saSignature
    val cond = options.condition
    val output = options.output

    val symTab = Map(
      "s" ->(0, TSample),
      "sa" ->(1, sas))
    val aggregationTable = Map(
      "v" ->(0, TVariant),
      "va" ->(1, vds.vaSignature),
      "s" ->(2, TSample),
      "sa" ->(3, vds.saSignature),
      "g" ->(4, TGenotype)
    )

    val ec = EvalContext(symTab, ("gs", EvalContext(aggregationTable)))


    val (header, fs) = if (cond.endsWith(".columns"))
      ExportTSV.parseColumnsFile(ec, cond, vds.sparkContext.hadoopConfiguration)
    else
      Parser.parseExportArgs(ec, cond)


    val a = ec.a
    val aggregatorA = ec.children("gs").a

    val sampleAggregations = Aggregators.buildSampleAggregations(vds, ec, "gs")

    hadoopDelete(output, state.hadoopConf, recursive = true)

    val sb = new StringBuilder()
    val lines = for (s <- vds.localSamples) yield {
      sb.clear()
      a(0) = vds.sampleIds(s)
      a(1) = vds.sampleAnnotations(s)
      sampleAggregations.foreach { arr =>
        arr(s).iterator
          .zip(ec.children("gs").aggregationFunctions.map(_._4).iterator)
          .foreach { case (value, j) =>
            aggregatorA(j) = value
          }
      }

      var first = true
      fs.foreach { f =>
        if (first)
          first = false
        else
          sb += '\t'
        sb.tsvAppend(f())
      }
      sb.result()
    }
    writeTable(output, hConf, lines, header)

    state
  }
}

package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.expr
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.annotations._
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

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val hConf = vds.sparkContext.hadoopConfiguration
    val sas = vds.metadata.sampleAnnotationSignatures
    val cond = options.condition
    val output = options.output

    val symTab = Map(
      "s" ->(0, expr.TSample),
      "sa" ->(1, vds.metadata.sampleAnnotationSignatures.toExprType))
    val a = new Array[Any](2)

    val (header, fs) = if (cond.endsWith(".columns"))
      ExportTSV.parseColumnsFile(symTab, a, cond, vds.sparkContext.hadoopConfiguration)
    else
      expr.Parser.parseExportArgs(symTab, a, cond)

    hadoopDelete(output, state.hadoopConf, recursive = true)

    val sb = new StringBuilder()
    val lines = for (s <- vds.localSamples) yield {
      sb.clear()
      a(0) = vds.sampleIds(s)
      a(1) = vds.metadata.sampleAnnotations(s).attrs
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

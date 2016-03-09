package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.expr
import org.broadinstitute.hail.expr.TVariant
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.annotations._
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.io.Source

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

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val vas = vds.metadata.vaSignatures
    val cond = options.condition
    val output = options.output

    val symTab = Map(
      "v" ->(0, TVariant),
      "va" ->(1, vds.metadata.vaSignatures.dType))
    val a = new Array[Any](2)

    val (header, fs) = if (cond.endsWith(".columns"))
      ExportTSV.parseColumnsFile(symTab, a, cond, vds.sparkContext.hadoopConfiguration)
    else
      expr.Parser.parseExportArgs(symTab, a, cond)

    hadoopDelete(output, state.hadoopConf, recursive = true)

    vds.variantsAndAnnotations
      .mapPartitions { it =>
        val sb = new StringBuilder()
        it.map { case (v, va) =>
          sb.clear()
          var first = true
          fs.foreach { f =>
            a(0) = v
            a(1) = va
            if (first)
              first = false
            else
              sb += '\t'
            sb.tsvAppend(f())
          }
          sb.result()
        }
      }.writeTable(output, header)

    state
  }
}

package org.broadinstitute.hail.driver

import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.io.TextExporter
import org.broadinstitute.hail.methods._
import org.kohsuke.args4j.{Option => Args4jOption}

object ExportSamples extends Command with TextExporter {

  class Options extends BaseOptions {

    @Args4jOption(required = true, name = "-o", aliases = Array("--output"),
      usage = "path of output tsv")
    var output: String = _

    @Args4jOption(required = true, name = "-c", aliases = Array("--columns"),
      usage = ".columns file, or comma-separated list of fields/computations to be printed to tsv")
    var condition: String = _

    @Args4jOption(required = false, name = "-t", aliases = Array("--types"),
      usage = "Write the types of parse expressions to a file at the given path")
    var typesFile: String = _
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
    val localGlobalAnnotation = vds.globalAnnotation

    val ec = Aggregators.sampleEC(vds)

    val (header, types, f) = Parser.parseNamedArgs(cond, ec)
    Option(options.typesFile).foreach { file =>
      val typeInfo = header
        .getOrElse(types.indices.map(i => s"_$i").toArray)
        .zip(types)
      exportTypes(file, state.hadoopConf, typeInfo)
    }

    val sampleAggregationOption = Aggregators.buildSampleAggregations(vds, ec)

    hConf.delete(output, recursive = true)

    val sb = new StringBuilder()
    val lines = for ((s, sa) <- vds.sampleIdsAndAnnotations) yield {
      sampleAggregationOption.foreach(f => f.apply(s))
      sb.clear()
      ec.setAll(localGlobalAnnotation, s, sa)
      f().foreachBetween(x => sb.append(x))(sb += '\t')
      sb.result()
    }

    hConf.writeTable(output, lines, header.map(_.mkString("\t")))

    state
  }

}

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

    val aggregationEC = EvalContext(Map(
      "v" -> (0, TVariant),
      "va" -> (1, vds.vaSignature),
      "s" -> (2, TSample),
      "sa" -> (3, vds.saSignature),
      "g" -> (4, TGenotype),
      "global" -> (5, vds.globalSignature)))

    val symTab = Map(
      "s" -> (0, TSample),
      "sa" -> (1, vds.saSignature),
      "global" -> (2, vds.globalSignature),
      "gs" -> (-1, BaseAggregable(aggregationEC, TGenotype)))

    val ec = EvalContext(symTab)
    ec.set(2, vds.globalAnnotation)
    aggregationEC.set(5, vds.globalAnnotation)

    val (header, fs) = if (cond.endsWith(".columns")) {
      val (h, functions) = Parser.parseColumnsFile(ec, cond, hConf)
      (Some(h), functions)
    }
    else
      Parser.parseExportArgs(cond, ec)

    Option(options.typesFile).foreach { file =>
      val typeInfo = header
        .getOrElse(fs.indices.map(i => s"_$i").toArray)
        .zip(fs.map(_._1))
      exportTypes(file, state.hadoopConf, typeInfo)
    }

    val sampleAggregationOption = Aggregators.buildSampleAggregations(vds, aggregationEC)

    hConf.delete(output, recursive = true)

    val sb = new StringBuilder()
    val lines = for ((s, sa) <- vds.sampleIdsAndAnnotations) yield {
      sb.clear()

      ec.setAll(s, sa)

      sampleAggregationOption.foreach(f => f.apply(s))

      var first = true
      fs.foreachBetween { case (t, f) =>
        sb.append(f().map(TableAnnotationImpex.exportAnnotation(_, t)).getOrElse("NA"))
      } { sb += '\t' }
      sb.result()
    }

    hConf.writeTable(output, lines, header.map(_.mkString("\t")))

    state
  }

}

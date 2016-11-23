package org.broadinstitute.hail.driver

import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.io.TextExporter
import org.broadinstitute.hail.methods._
import org.kohsuke.args4j.{Option => Args4jOption}

object ExportVariants extends Command with TextExporter {

  class Options extends BaseOptions {

    @Args4jOption(required = true, name = "-o", aliases = Array("--output"),
      usage = "path of output tsv")
    var output: String = _

    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = ".columns file, or comma-separated list of fields/computations to be printed to tsv")
    var condition: String = _

    @Args4jOption(required = false, name = "-t", aliases = Array("--types"),
      usage = "Write the types of parse expressions to a file at the given path")
    var typesFile: String = _

  }

  def newOptions = new Options

  def name = "exportvariants"

  def description = "Export list of variant information to tsv"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val vas = vds.vaSignature
    val hConf = vds.sparkContext.hadoopConfiguration
    val cond = options.condition
    val output = options.output

    val localGlobalAnnotations = vds.globalAnnotation
    val ec = Aggregators.variantEC(vds)

    val (header, types, f) = Parser.parseNamedArgs(cond, ec)

    Option(options.typesFile).foreach { file =>
      val typeInfo = header
        .getOrElse(types.indices.map(i => s"_$i").toArray)
        .zip(types)
      exportTypes(file, state.hadoopConf, typeInfo)
    }

    val variantAggregations = Aggregators.buildVariantAggregations(vds, ec)

    state.hadoopConf.delete(output, recursive = true)

    vds.rdd
      .mapPartitions { it =>
        val sb = new StringBuilder()
        it.map { case (v, (va, gs)) =>
          variantAggregations.foreach { f => f(v, va, gs) }
          ec.setAll(localGlobalAnnotations, v, va)
          sb.clear()
          f().foreachBetween(x => sb.append(x))(sb += '\t')
          sb.result()
        }
      }.writeTable(output, header.map(_.mkString("\t")))

    state
  }
}

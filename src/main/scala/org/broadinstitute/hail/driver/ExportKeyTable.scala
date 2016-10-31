package org.broadinstitute.hail.driver

import org.broadinstitute.hail.expr.{EvalContext, _}
import org.broadinstitute.hail.io.TextExporter
import org.broadinstitute.hail.utils._
import org.kohsuke.args4j.{Option => Args4jOption}

object ExportKeyTable extends Command with TextExporter {

  class Options extends BaseOptions {

    @Args4jOption(required = true, name = "-o", aliases = Array("--output"),
      usage = "path of output tsv")
    var output: String = _

    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = ".columns file, or comma-separated list of fields/computations to be printed to tsv")
    var condition: String = _

    @Args4jOption(required = true, name = "-n", aliases = Array("--name"),
      usage = "name of key table to be printed to tsv")
    var name: String = _

    @Args4jOption(required = false, name = "-t", aliases = Array("--types"),
      usage = "Write the types of parse expressions to a file at the given path")
    var typesFile: String = _

  }

  def newOptions = new Options

  def name = "ktexport"

  def description = "Export information from key table to tsv"

  def supportsMultiallelic = true

  def requiresVDS = false

  override def hidden = true

  def run(state: State, options: Options): State = {

    val kt = state.ktEnv.get(options.name) match {
      case Some(newKT) =>
        newKT
      case None =>
        fatal("no such key table $name in environment")
    }

    val cond = options.condition
    val output = options.output

    val symTab = Map(
      kt.keyIdentifier -> (0, kt.keySignature),
      kt.valueIdentifier -> (1, kt.valueSignature),
      "global" -> (2, state.vds.globalSignature)
    )

    val ec = EvalContext(symTab)

    val (header, types, f) = Parser.parseExportArgs(cond, ec)

    Option(options.typesFile).foreach { file =>
      val typeInfo = header
        .getOrElse(types.indices.map(i => s"_$i").toArray)
        .zip(types)
      exportTypes(file, state.hadoopConf, typeInfo)
    }

    state.hadoopConf.delete(output, recursive = true)

    kt.rdd
      .mapPartitions { it =>
        val sb = new StringBuilder()
        it.map { case (k, v) =>
          sb.clear()

          ec.setAll(k, v)

          f().foreachBetween(x => sb.append(x))(sb += '\t')
          sb.result()
        }
      }.writeTable(output, header.map(_.mkString("\t")))

    state
  }
}


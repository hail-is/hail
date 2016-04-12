package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods._
import org.kohsuke.args4j.{Option => Args4jOption}
import scala.collection.mutable.ArrayBuffer

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

  override def supportsMultiallelic = true

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val vas = vds.vaSignature
    val cond = options.condition
    val output = options.output

    val symTab = Map(
      "v" ->(0, TVariant),
      "va" ->(1, vas))
    val a = new ArrayBuffer[Any]()
    for (_ <- symTab)
      a += null

    val (header, fs) = if (cond.endsWith(".columns"))
      ExportTSV.parseColumnsFile(symTab, a, cond, vds.sparkContext.hadoopConfiguration)
    else
      Parser.parseExportArgs(cond, symTab, a)

    hadoopDelete(output, state.hadoopConf, recursive = true)

    vds.variantsAndAnnotations
      .mapPartitions { it =>
        val sb = new StringBuilder()
        it.map { case (v, va) =>
          sb.clear()
          a(0) = v
          a(1) = va
          fs.iterator.foreachBetween { f => sb.tsvAppend(f()) }(() => sb.append("\t"))
          sb.result()
        }
      }.writeTable(output, header)

    state
  }
}

package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.variant._
import org.kohsuke.args4j.{Option => Args4jOption}
import scala.collection.mutable.ArrayBuffer

object ExportGenotypes extends Command {

  class Options extends BaseOptions {

    @Args4jOption(required = true, name = "-o", aliases = Array("--output"),
      usage = "path of output tsv")
    var output: String = _

    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = ".columns file, or comma-separated list of fields/computations to be printed to tsv")
    var condition: String = _

    @Args4jOption(name = "--print-ref", usage = "print reference genotypes")
    var printRef: Boolean = false

    @Args4jOption(name = "--print-missing", usage = "print reference genotypes")
    var printMissing: Boolean = _

  }

  def newOptions = new Options

  def name = "exportgenotypes"

  def description = "Export list of sample-variant information to tsv"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val sc = vds.sparkContext
    val cond = options.condition
    val output = options.output
    val vas = vds.vaSignature
    val sas = vds.saSignature

    val symTab = Map(
      "v" ->(0, TVariant),
      "va" ->(1, vas),
      "s" ->(2, TSample),
      "sa" ->(3, sas),
      "g" ->(4, TGenotype))

    val ec = EvalContext(symTab)

    val (header, fs) = if (cond.endsWith(".columns"))
      ExportTSV.parseColumnsFile(ec, cond, sc.hadoopConfiguration)
    else
      Parser.parseExportArgs(cond, ec)

    val a = ec.a

    hadoopDelete(output, state.hadoopConf, recursive = true)

    val sampleIdsBc = sc.broadcast(vds.sampleIds)
    val sampleAnnotationsBc = sc.broadcast(vds.sampleAnnotations)

    val localPrintRef = options.printRef
    val localPrintMissing = options.printMissing

    val filterF: Genotype => Boolean =
      g => (!g.isHomRef || localPrintRef) && (!g.isNotCalled || localPrintMissing)

    val lines = vds.mapPartitionsWithAll { it =>
      val sb = new StringBuilder()
      it
        .filter { case (v, va, s, sa, g) => filterF(g) }
        .map { case (v, va, s, sa, g) =>
          a(0) = v
          a(1) = va
          a(2) = s
          a(3) = sa
          a(4) = g
          sb.clear()
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
    }.writeTable(output, header)

    state
  }
}

package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.expr
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.annotations._
import org.kohsuke.args4j.{Option => Args4jOption}

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

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val sc = vds.sparkContext
    val cond = options.condition
    val output = options.output
    val vas = vds.metadata.variantAnnotationSignatures
    val sas = vds.metadata.sampleAnnotationSignatures

    val symTab = Map(
      "v" ->(0, expr.TVariant),
      "va" ->(1, vds.metadata.variantAnnotationSignatures.dType),
      "s" ->(2, expr.TSample),
      "sa" ->(3, vds.metadata.sampleAnnotationSignatures.dType),
      "g" ->(4, expr.TGenotype))
    val a = new Array[Any](5)

    val (header, fs) = if (cond.endsWith(".columns"))
      ExportTSV.parseColumnsFile(symTab, a, cond, sc.hadoopConfiguration)
    else
      expr.Parser.parseExportArgs(symTab, a, cond)

    hadoopDelete(output, state.hadoopConf, recursive = true)

    val sampleIdsBc = sc.broadcast(vds.sampleIds)
    val sampleAnnotationsBc = sc.broadcast(vds.metadata.sampleAnnotations)

    val localPrintRef = options.printRef
    val localPrintMissing = options.printMissing

    val filterF: Genotype => Boolean =
      g => (!g.isHomRef || localPrintRef) && (!g.isNotCalled || localPrintMissing)

    val lines = vds.mapPartitionsWithAll { it =>
      val sb = new StringBuilder()
      it
        .filter { case (v, va, s, g) => filterF(g) }
        .map { case (v, va, s, g) =>
          a(0) = v
          a(1) = va
          a(2) = sampleIdsBc.value(s)
          a(3) = sampleAnnotationsBc.value(s)
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

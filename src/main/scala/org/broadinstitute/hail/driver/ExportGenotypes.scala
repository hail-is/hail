package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.variant._
import org.kohsuke.args4j.{Option => Args4jOption}

object ExportGenotypes extends Command {

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
      "v" -> (0, TVariant),
      "va" -> (1, vas),
      "s" -> (2, TSample),
      "sa" -> (3, sas),
      "g" -> (4, TGenotype))

    val ec = EvalContext(symTab)

    val (header, parseResults) = if (cond.endsWith(".columns")) {
      val (h, functions) = ExportTSV.parseColumnsFile(ec, cond, sc.hadoopConfiguration)
      (Some(h), functions)
    }
    else
      Parser.parseExportArgs(cond, ec)

    Option(options.typesFile).foreach { file =>
      val typeInfo = header
        .getOrElse(parseResults.indices.map(i => s"_$i").toArray)
        .zip(parseResults.map(_._1))
      ExportTSV.exportTypes(file, state.hadoopConf, typeInfo)
    }

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
          ec.setAll(v, va, s, sa, g)
          sb.clear()
          parseResults.foreachBetween { case (t, f) => sb.append(f().map(t.str).getOrElse("NA")) }(() => sb += '\t')
          sb.result()
        }
    }.writeTable(output, header.map(_.mkString("\t")))

    state
  }
}

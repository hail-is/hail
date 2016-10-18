package org.broadinstitute.hail.driver

import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.io.TextExporter
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.variant._
import org.kohsuke.args4j.{Option => Args4jOption}

object ExportGenotypes extends Command with TextExporter {

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
      "g" -> (4, TGenotype),
      "global" -> (5, vds.globalSignature))

    val ec = EvalContext(symTab)
    ec.set(5, vds.globalAnnotation)
    val (header, ts, f) = Parser.parseExportArgs(cond, ec)

    Option(options.typesFile).foreach { file =>
      val typeInfo = header
        .getOrElse(ts.indices.map(i => s"_$i").toArray)
        .zip(ts)
      exportTypes(file, state.hadoopConf, typeInfo)
    }

    state.hadoopConf.delete(output, recursive = true)

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

          f().foreachBetween(x => sb.append(x))(sb += '\t')
          sb.result()
        }
    }.writeTable(output, header.map(_.mkString("\t")))

    state
  }
}

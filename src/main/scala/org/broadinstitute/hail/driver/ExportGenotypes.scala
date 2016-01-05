package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
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
  }

  def newOptions = new Options

  def name = "exportgenotypes"

  def description = "Export list of sample-variant information to tsv"

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val cond = options.condition
    val output = options.output
    val vas: AnnotationSignatures = vds.metadata.variantAnnotationSignatures
    val sas: AnnotationSignatures = vds.metadata.sampleAnnotationSignatures

    val (header, fields) = ExportTSV.parseExpression(cond)

    val makeString: ((Variant, AnnotationData) =>
      ((Int, Genotype) => String)) = {
      val cf = new ExportGenotypeEvaluator(fields, vds.metadata)
      cf.typeCheck()
      cf.apply
    }

    val stringVDS = vds.mapValuesWithPartialApplication(
      (v: Variant, va: AnnotationData) =>
        (s: Int, g: Genotype) =>
          makeString(v, va)(s, g))

    writeTextFile(output + ".header", state.hadoopConf) { s =>
      s.write(header)
      s.write("\n")
    }

    hadoopDelete(output, state.hadoopConf, recursive = true)

    stringVDS.rdd
      .flatMap { case (v, va, strings) => strings }
      .saveAsTextFile(output)

    state
  }
}

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
      usage = "Comma-separated list of fields to be printed to tsv")
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
    val sa = vds.metadata.sampleAnnotations
    val ids = vds.sampleIds

    val makeString: ((Variant, AnnotationData) =>
      ((Int, Genotype) => String)) = {
      val cf = new ExportGenotypeEvaluator(options.condition, vas, sas, sa, ids)
      cf.typeCheck()
      cf.apply
    }

    val stringVDS = vds.mapValuesWithPartialApplication(
      (v: Variant, va: AnnotationData) =>
        (s: Int, g: Genotype) =>
          makeString(v, va)(s, g))

    // FIXME add additional command parsing functionality
    val variantRegex =
      """v\.(\w+)""".r
    val sampleRegex = """s\.(\w+)""".r
    val topLevelSampleAnnoRegex = """sa\.(\w+)""".r
    val topLevelVariantAnnoRegex = """va\.(\w+)""".r
    val samplePrintMapRegex = """sa\.(\w+)\.all""".r
    val variantPrintMapRegex = """va\.(\w+)\.all""".r
    val annoRegex = """\wa\.(.+)""".r
    def mapColumnNames(input: String): String = {
      input match {
        case "v" => "Variant"
        case "s" => "Sample"
        case "va" =>
          fatal("parse error in condition: cannot print 'va', choose a group or value in annotations")
        case "sa" =>
          fatal("parse error in condition: cannot print 'sa', choose a group or value in annotations")
        case variantRegex(x) => x
        case sampleRegex(x) => x
        case topLevelSampleAnnoRegex(x) =>
          if (sas.maps.contains(x)) {
            val keys = sas.maps(x).keys.toArray.sorted
            if (keys.isEmpty) x else s"$x:" + keys.mkString(";")
          }
          else x
        case topLevelVariantAnnoRegex(x) =>
          if (vas.maps.contains(x)) {
            val keys = vas.maps(x).keys.toArray.sorted
            if (keys.isEmpty) x else s"$x:" + keys.mkString(";")
          }
          else x
        case samplePrintMapRegex(x) =>
          val keys = sas.maps(x).keys
          if (keys.isEmpty) x else keys.mkString("\t")
        case variantPrintMapRegex(x) =>
          val keys = vas.maps(x).keys
          if (keys.isEmpty) x else keys.mkString("\t")
        case annoRegex(x) => x
        case _ => input
      }
    }

    writeTextFile(output + ".header", state.hadoopConf) { s =>
      s.write(cond.split(",").map(_.split("\\.").last).mkString("\t"))
      s.write("\n")
    }

    hadoopDelete(output, state.hadoopConf, recursive = true)

    stringVDS.rdd
      .flatMap { case (v, va, strings) => strings }
      .saveAsTextFile(output)

    state
  }
}

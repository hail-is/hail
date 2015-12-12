package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.annotations._
import org.kohsuke.args4j.{Option => Args4jOption}

object ExportSamples extends Command {

  class Options extends BaseOptions {

    @Args4jOption(required = true, name = "-o", aliases = Array("--output"),
      usage = "path of output tsv")
    var output: String = _

    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = "Comma-separated list of fields to be printed to tsv")
    var condition: String = _

    @Args4jOption(required = false, name = "--missing",
      usage = "Format of missing values (Default: 'NA')")
    var missing = "NA"
  }

  def newOptions = new Options

  def name = "exportsamples"

  def description = "Export list of sample information to tsv"

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val cond = options.condition

    val output = options.output

    val sas = vds.metadata.sampleAnnotationSignatures
    val makeString: (Sample, Annotations[String]) => String = {
      try {
        val ese = new ExportSamplesEvaluator(cond, sas, options.missing)
        ese.typeCheck()
        ese.apply
      } catch {
        case e: scala.tools.reflect.ToolBoxError =>
          /* e.message looks like:
             reflective compilation has failed:

             ';' expected but '.' found. */
          fatal("parse error in condition: " + e.message.split("\n").last)
      }
    }

    // FIXME add additional command parsing functionality
    val sampleRegex = """s\.(\w+)""".r
    val topLevelAnnoRegex = """sa\.(\w+)""".r
    val printMapRegex = """sa\.(\w+)\.all""".r
    val annoRegex = """sa\.(.+)""".r
    def mapColumnNames(input: String): String = {
      input match {
        case "v" => "Sample"
        case "sa" =>
          fatal("parse error in condition: cannot print 'sa', choose a group or value in annotations")
        case sampleRegex(x) => x
        case topLevelAnnoRegex(x) =>
          if (sas.maps.contains(x)) {
            val keys = sas.maps(x).keys.toArray.sorted
            if (keys.isEmpty) x else s"$x:" + keys.mkString(";")
          }
          else x
        case printMapRegex(x) =>
          val keys = sas.maps(x).keys
          if (keys.isEmpty) x else keys.mkString("\t")
        case annoRegex(x) => x
        case _ => input
      }
    }

    writeTextFile(output + ".header", state.hadoopConf) { s =>
      s.write(cond.split(",").map(_.split("\\.").last).mkString(";"))
      s.write("\n")
    }

    hadoopDelete(output, state.hadoopConf, recursive = true)

    vds.sparkContext.parallelize(vds.sampleIds.map(Sample).zip(vds.metadata.sampleAnnotations))
      .map { case (s, sa) => makeString(s, sa) }
      .saveAsTextFile(output)

    state
  }
}

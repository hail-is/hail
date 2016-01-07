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
      usage = ".columns file, or comma-separated list of fields/computations to be printed to tsv")
    var condition: String = _
  }

  def newOptions = new Options

  def name = "exportsamples"

  def description = "Export list of sample information to tsv"

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val sas = vds.metadata.sampleAnnotationSignatures
    val cond = options.condition
    val output = options.output

    val (header, fields) = if (cond.endsWith(".columns"))
      ExportTSV.parseColumnsFile(cond, state.hadoopConf)
    else
      ExportTSV.parseExpression(cond)

    val makeString: (Sample, Annotations[String]) => String = {
      val ese = new ExportSamplesEvaluator(fields, sas)
      ese.typeCheck()
      ese.apply
    }

    // FIXME add additional command parsing functionality

    header match {
      case Some(str) =>
        writeTextFile(output + ".header", state.hadoopConf) { s =>
          s.write(str)
          s.write("\n")
        }
    }

    hadoopDelete(output, state.hadoopConf, recursive = true)

    vds.sparkContext.parallelize(vds.sampleIds.map(Sample).zip(vds.metadata.sampleAnnotations))
      .map { case (s, sa) => makeString(s, sa) }
      .saveAsTextFile(output)

    state
  }
}

package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.annotations._
import org.kohsuke.args4j.{Option => Args4jOption}

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

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val vas = vds.metadata.variantAnnotationSignatures
    val cond = options.condition
    val output = options.output

    val (header, fields) = if (cond.endsWith(".columns"))
      ExportTSV.parseColumnsFile(cond, state.hadoopConf)
    else
      ExportTSV.parseExpression(cond)

    val makeString: (Variant, Annotations[String]) => String = {
      val eve = new ExportVariantsEvaluator(fields, vas)
      eve.typeCheck()
      eve.apply
    }

    hadoopDelete(output, state.hadoopConf, recursive = true)

    vds.variantsAndAnnotations
      .map { case (v, va) => makeString(v, va) }
      .writeTable(output, header)

    state
  }


}

package org.broadinstitute.hail.driver

import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.io.annotators.GlobalTableAnnotator
import org.kohsuke.args4j.{Option => Args4jOption}

object AnnotateGlobalTable extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-i", aliases = Array("--input"),
      usage = "Path to file")
    var input: String = _

    @Args4jOption(required = false, name = "-t", aliases = Array("--types"),
      usage = "Define types of fields in file")
    var types: String = ""

    @Args4jOption(required = true, name = "-r", aliases = Array("--root"),
      usage = "Argument is a period-delimited path starting with `global'")
    var root: String = _

    @Args4jOption(required = false, name = "-m", aliases = Array("--missing"),
      usage = "Specify identifier to be treated as missing")
    var missing: String = "NA"

    @Args4jOption(required = false, name = "-d", aliases = Array("--delimiter"),
      usage = "Field delimiter regex")
    var delimiter: String = "\\t"
  }

  def newOptions = new Options

  def name = "annotateglobal table"

  def description = "Annotate global table from a text file with multiple columns"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val path = Parser.parseAnnotationRoot(options.root, Annotation.GLOBAL_HEAD)

    val (result, signature) = GlobalTableAnnotator(options.input, state.hadoopConf,
      Parser.parseAnnotationTypes(options.types), options.missing, options.delimiter)


    val (newGlobalSig, inserter) = vds.insertGlobal(signature, path)

    state.copy(vds = vds.copy(
      globalAnnotation = inserter(vds.globalAnnotation, Some(result)),
      globalSignature = newGlobalSig))
  }
}


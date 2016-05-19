package org.broadinstitute.hail.driver

import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.io.annotators.SampleTableAnnotator
import org.kohsuke.args4j.{Option => Args4jOption}

object AnnotateSamplesTable extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-i", aliases = Array("--input"),
      usage = "TSV file path")
    var input: String = _

    @Args4jOption(name = "-s", aliases = Array("--sampleheader"),
      usage = "Identify the name of the column containing the sample IDs")
    var sampleCol: String = "Sample"

    @Args4jOption(required = false, name = "-t", aliases = Array("--types"),
      usage = "Define types of fields in annotations files")
    var types: String = ""

    @Args4jOption(required = true, name = "-r", aliases = Array("--root"),
      usage = "Argument is a period-delimited path starting with `sa'")
    var root: String = _

    @Args4jOption(required = false, name = "-m", aliases = Array("--missing"),
      usage = "Specify identifier to be treated as missing")
    var missing: String = "NA"

    @Args4jOption(required = false, name = "-d", aliases = Array("--delimiter"),
      usage = "Field delimiter regex")
    var delimiter: String = "\\t"
  }

  def newOptions = new Options

  def name = "annotatesamples table"

  def description = "Annotate samples with a text table"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val input = options.input

    val (m, signature) = SampleTableAnnotator(input, options.sampleCol,
      Parser.parseAnnotationTypes(options.types),
      options.missing,
      state.hadoopConf, options.delimiter)
    val annotated = vds.annotateSamples(m, signature,
      Parser.parseAnnotationRoot(options.root, Annotation.SAMPLE_HEAD))
    state.copy(vds = annotated)
  }
}

package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.io.annotators.{SampleFamAnnotator, SampleTSVAnnotator}
import org.kohsuke.args4j.{Option => Args4jOption}
import org.broadinstitute.hail.expr._

object AnnotateSamplesFam extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-i", aliases = Array("--input"),
      usage = ".fam file path")
    var input: String = _

    @Args4jOption(required = false, name = "-q", aliases = Array("--quantpheno"),
      usage = "Quantitative phenotype flag")
    var isQuantitative: Boolean = false

    @Args4jOption(required = false, name = "-d", aliases = Array("--delimiter"),
      usage = "Field delimiter")
    var delimiter: String = "\t"

    @Args4jOption(required = false, name = "-r", aliases = Array("--root"),
      usage = "Annotation root, a period-delimited path starting with `sa'")
    var root: String = "sa.fam"

    @Args4jOption(required = false, name = "-m", aliases = Array("--missing"),
      usage = "Identifier to be treated as missing (for case-control, in addition to `0', `-9', and non-numeric)")
    var missing: String = "NA"
  }

  def newOptions = new Options

  def name = "annotatesamples fam"

  def description = "Annotate samples with .fam file: famID, patID, matID, isMale, and either isCase or qPheno"

  override def supportsMultiallelic = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val input = options.input

    fatalIf(!input.endsWith(".fam"), "input file must end in .fam")

    val delimiter = options.delimiter
    val isQuantitative = options.isQuantitative

    val (m, signature) = SampleFamAnnotator(input, delimiter, isQuantitative, options.missing, state.hadoopConf)
    val annotated = vds.annotateSamples(m, signature, Parser.parseAnnotationRoot(options.root, "sa"))
    state.copy(vds = annotated)
  }
}

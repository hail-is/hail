package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.io.annotators.{SampleFamAnnotator, SampleTSVAnnotator}
import org.kohsuke.args4j.{Option => Args4jOption}

object AnnotateSamplesFam extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-i", aliases = Array("--input"),
      usage = ".fam file path")
    var input: String = _

    @Args4jOption(required = false, name = "-q", aliases = Array("--quantpheno"),
      usage = "Use this flag if phenotype is quanitative")
    var isQuantitative: Boolean = false

    @Args4jOption(required = false, name = "-d", aliases = Array("--delimiter"),
      usage = "Field delimiter, default is \\t") // FIXME: specify some options here and in documentation
    var delimiter: String = "\t"

    @Args4jOption(required = false, name = "-r", aliases = Array("--root"),
      usage = "Argument is a period-delimited path starting with `sa', default is `sa.fam'")
    var root: String = "sa.fam"
  }

  def newOptions = new Options

  def name = "annotatesamples fam"

  def description = "Annotate samples with .fam file"

  override def supportsMultiallelic = true

  def parseRoot(s: String): List[String] = {
    val split = s.split("\\.").toList
    fatalIf(split.isEmpty || split.head != "sa", s"Root must start with `sa.', got `$s'")
    split.tail
  }

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val input = options.input

    fatalIf(!input.endsWith(".fam"), "input file must end in .fam")

    val delimiter = options.delimiter
    val isQuantitative = options.isQuantitative

    val (m, signature) = SampleFamAnnotator(input, delimiter, isQuantitative, state.hadoopConf)
    val annotated = vds.annotateSamples(m, signature, parseRoot(options.root))
    state.copy(vds = annotated)
  }
}

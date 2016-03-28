package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.{Annotation, Inserter}
import org.broadinstitute.hail.expr
import org.broadinstitute.hail.io.annotators.SampleTSVAnnotator
import org.broadinstitute.hail.variant.Sample
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable

object AnnotateSamplesTSV extends Command {

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
      usage = "Specify identifiers to be treated as missing")
    var missing: String = "NA"
  }

  def newOptions = new Options

  def name = "annotatesamples/tsv"

  def description = "Annotate samples in current dataset"

  override def hidden = true

  override def supportsMultiallelic = true

  def parseRoot(s: String): List[String] = {
    val split = s.split("""\.""").toList
    fatalIf(split.isEmpty || split.head != "sa", s"invalid root '$s': expect 'sa.<path[.path2...]>'")
    split.tail
  }

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val input = options.input

    val (m, signature) = SampleTSVAnnotator(input, options.sampleCol,
      AnnotateVariantsTSV.parseTypeMap(options.types), options.missing,
      vds.sparkContext.hadoopConfiguration)
    val annotated = vds.annotateSamples(m, signature, parseRoot(options.root))
    state.copy(vds = annotated)
  }
}

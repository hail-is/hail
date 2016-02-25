package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotations
import org.broadinstitute.hail.io.annotators.SampleTSVAnnotator
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.variant.Sample
import org.kohsuke.args4j.{Option => Args4jOption}

object AnnotateSamples extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = "Annotation file path")
    var condition: String = _

    @Args4jOption(name = "-s", aliases = Array("--sampleheader"),
      usage = "Identify the name of the column containing the sample IDs (default: 'Sample')")
    var sampleCol: String = "Sample"

    @Args4jOption(required = false, name = "-t", aliases = Array("--types"),
      usage = "Define types of fields in annotations files")
    var types: String = _

    @Args4jOption(required = false, name = "-r", aliases = Array("--root"),
      usage = "Place annotations in the path 'sa.<root>.<field>, or sa.<field> if unspecified'")
    var root: String = _

    @Args4jOption(required = false, name = "-m", aliases = Array("--missing"),
      usage = "Specify identifiers to be treated as missing (default: 'NA')")
    var missingIdentifiers: String = "NA"
  }

  def newOptions = new Options

  def name = "annotatesamples"

  def description = "Annotate samples in current dataset"

  override def supportsMultiallelic = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val cond = options.condition

    val root = options.root match {
      case null => null
      case r if r.startsWith("sa.") => r.substring(3)
      case error => fatal(s"invalid root '$error': expect 'sa.<path[.path2...]>'")
    }

    val stripped = hadoopStripCodec(cond, state.sc.hadoopConfiguration)
    val annotator = stripped match {
      case tsv if tsv.endsWith(".tsv") =>
        new SampleTSVAnnotator(cond, options.sampleCol,
          AnnotateVariants.parseTypeMap(options.types),
          AnnotateVariants.parseMissing(options.missingIdentifiers),
          root, vds.sparkContext.hadoopConfiguration)
      case _ => fatal(s"unknown file type '$cond'.  Specify a .tsv file")
    }
    state.copy(vds = vds.annotateSamples(annotator))
  }
}

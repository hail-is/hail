package org.broadinstitute.hail.driver

import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.io.annotators._
import org.kohsuke.args4j.{Option => Args4jOption}

object AnnotateVariantsBed extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-i", aliases = Array("--input"),
      usage = "Bed file path")
    var input: String = _

    @Args4jOption(required = true, name = "-r", aliases = Array("--root"),
      usage = "Period-delimited path starting with `va'")
    var root: String = _
  }

  def newOptions = new Options

  def name = "annotatevariants bed"

  def description = "Annotate variants with UCSC BED file"

  def supportsMultiallelic = false

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val (t, o) = BedAnnotator(options.input, state.hadoopConf)
    state.copy(
      vds = state.vds.annotateIntervals(t, o,
        Parser.parseAnnotationRoot(options.root, Annotation.VARIANT_HEAD)))
  }
}

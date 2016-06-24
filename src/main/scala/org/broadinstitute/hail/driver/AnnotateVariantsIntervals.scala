package org.broadinstitute.hail.driver

import org.broadinstitute.hail.io.annotators._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.kohsuke.args4j.{Option => Args4jOption}

object AnnotateVariantsIntervals extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-i", aliases = Array("--input"),
      usage = "Interval file path")
    var input: String = _

    @Args4jOption(required = true, name = "-r", aliases = Array("--root"),
      usage = "Period-delimited path starting with `va'")
    var root: String = _
  }

  def newOptions = new Options

  def name = "annotatevariants intervals"

  def description = "Annotate variants with interval list"

  def supportsMultiallelic = false

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val (iList, signature) = IntervalListAnnotator(options.input, state.hadoopConf)
    val annotated = vds.annotateIntervals(iList, signature,
      Parser.parseAnnotationRoot(options.root, Annotation.VARIANT_HEAD))
    state.copy(vds = annotated)
  }
}

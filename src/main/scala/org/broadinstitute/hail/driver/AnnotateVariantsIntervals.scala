package org.broadinstitute.hail.driver

import org.broadinstitute.hail.io.annotators._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.kohsuke.args4j.{Option => Args4jOption}

object AnnotateVariantsIntervals extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "-a", aliases = Array("--all"),
      usage = "When annotating with a value, annotate all values as a set")
    var all: Boolean = false

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

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val path = Parser.parseAnnotationRoot(options.root, Annotation.VARIANT_HEAD)
    val vds = state.vds
    val all = options.all
    val annotated = IntervalListAnnotator(options.input, state.hadoopConf) match {
      case (is, Some((m, t))) =>
        vds.annotateIntervals(is, m, t, all = all, path)

      case (is, None) =>
        vds.annotateIntervals(is, path)
    }

    state.copy(vds = annotated)
  }
}

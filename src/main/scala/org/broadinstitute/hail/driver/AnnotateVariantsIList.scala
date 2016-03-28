package org.broadinstitute.hail.driver

import org.broadinstitute.hail.io.annotators._
import org.kohsuke.args4j.{Option => Args4jOption}

object AnnotateVariantsIList extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-i", aliases = Array("--input"),
      usage = "Interval file path")
    var input: String = _

    @Args4jOption(required = true, name = "-r", aliases = Array("--root"),
      usage = "Period-delimited path starting with `va'")
    var root: String = _
  }

  def newOptions = new Options

  def name = "annotatevariants/intervals"

  override def hidden = true

  def description = "Annotate variants in current dataset"

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val cond = options.input

    val conf = state.sc.hadoopConfiguration

    val (iList, signature) = IntervalListAnnotator(cond, conf)
    val annotated = vds.annotateInvervals(iList, signature, AnnotateVariantsTSV.parseRoot(options.root))

    state.copy(vds = annotated)
  }
}

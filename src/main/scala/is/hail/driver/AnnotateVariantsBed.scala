package is.hail.driver

import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.io.annotators._
import org.kohsuke.args4j.{Option => Args4jOption}

object AnnotateVariantsBed extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "-a", aliases = Array("--all"),
      usage = "When annotating with a value, annotate all values as a set")
    var all: Boolean = false

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

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val all = options.all
    val path = Parser.parseAnnotationRoot(options.root, Annotation.VARIANT_HEAD)
    val newVDS = BedAnnotator(options.input, state.hadoopConf) match {
      case (is, None) =>
        vds.annotateIntervals(is, path)

      case (is, Some((t, m))) =>
        vds.annotateIntervals(is, t, m, all = all, path)
    }

    state.copy(vds = newVDS)
  }
}

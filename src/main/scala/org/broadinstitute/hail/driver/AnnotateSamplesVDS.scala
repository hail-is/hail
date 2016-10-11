package org.broadinstitute.hail.driver

import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr.{EvalContext, _}
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant.VariantSampleMatrix
import org.kohsuke.args4j.{Option => Args4jOption}

object AnnotateSamplesVDS extends Command with JoinAnnotator {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-i", aliases = Array("--input"),
      usage = "VDS file path")
    var input: String = _

    @Args4jOption(required = false, name = "-r", aliases = Array("--root"),
      usage = "Period-delimited path starting with `sa' (this argument or --code required)")
    var root: String = _

    @Args4jOption(required = false, name = "-c", aliases = Array("--code"),
      usage = "Use annotation expressions to join with the table (this argument or --root required)")
    var code: String = _
  }

  def newOptions = new Options

  def name = "annotatesamples vds"

  def description = "Annotate samples with VDS file"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val (expr, code) = (Option(options.code), Option(options.root)) match {
      case (Some(c), None) => (true, c)
      case (None, Some(r)) => (false, r)
      case _ => fatal("this module requires one of `--root' or `--code', but not both")
    }

    val otherVds = VariantSampleMatrix.read(state.sqlContext, options.input, skipVariants = true)

    val (finalType, inserter): (Type, (Annotation, Option[Annotation]) => Annotation) = if (expr) {
      val ec = EvalContext(Map(
        "sa" -> (0, vds.saSignature),
        "vds" -> (1, otherVds.saSignature)))
      buildInserter(code, vds.saSignature, ec, Annotation.SAMPLE_HEAD)
    } else vds.insertVA(otherVds.vaSignature, Parser.parseAnnotationRoot(code, Annotation.SAMPLE_HEAD))

    state.copy(vds = vds
      .annotateSamples(otherVds.sampleIdsAndAnnotations.toMap.get(_), finalType, inserter))
  }
}

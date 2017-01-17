package org.broadinstitute.hail.driver

import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr.{EvalContext, _}
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant.{VariantDataset, VariantSampleMatrix}
import org.kohsuke.args4j.{Option => Args4jOption}

object AnnotateSamplesVDS extends Command with JoinAnnotator {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-n", aliases = Array("--name"),
      usage = "name of VDS in environment")
    var name: String = _

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

  def annotate(vds: VariantDataset, other: VariantDataset, code: String, root: String): VariantDataset = {
    if (!((code != null) ^ (root != null)))
      fatal("either `--code' or `--root' required, but not both")

    val (finalType, inserter): (Type, (Annotation, Option[Annotation]) => Annotation) = if (code != null) {
      val ec = EvalContext(Map(
        "sa" -> (0, vds.saSignature),
        "vds" -> (1, other.saSignature)))
      buildInserter(code, vds.saSignature, ec, Annotation.SAMPLE_HEAD)
    } else vds.insertSA(other.saSignature, Parser.parseAnnotationRoot(root, Annotation.SAMPLE_HEAD))

    val m = other.sampleIdsAndAnnotations.toMap
    vds
      .annotateSamples(m.get _, finalType, inserter)
  }

  def run(state: State, options: Options): State = {
    val other = state.env.getOrElse(options.name, fatal(s"no VDS found with name `${options.name}'"))
    state.copy(vds = annotate(state.vds, other, options.code, options.root))
  }
}

package org.broadinstitute.hail.driver

import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr.{EvalContext, _}
import org.kohsuke.args4j.{Option => Args4jOption}

object AnnotateVariantsVDS extends Command with JoinAnnotator {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-i", aliases = Array("--input"),
      usage = "VDS file path")
    var input: String = _

    @Args4jOption(required = false, name = "-r", aliases = Array("--root"),
      usage = "Period-delimited path starting with `va' (this argument or --code required)")
    var root: String = _

    @Args4jOption(required = false, name = "-c", aliases = Array("--code"),
      usage = "Use annotation expressions to join with the table (this argument or --root required)")
    var code: String = _

    @Args4jOption(required = false, name = "--split",
      usage = "split multiallelic variants in the input VDS")
    var split: Boolean = _

  }

  def newOptions = new Options

  def name = "annotatevariants vds"

  def description = "Annotate variants with VDS file"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val (expr, code) = (Option(options.code), Option(options.root)) match {
      case (Some(c), None) => (true, c)
      case (None, Some(r)) => (false, r)
      case _ => fatal("this module requires one of `--root' or `--code', but not both")
    }

    val otherVds = {
      val s = Read.run(state, Array("--skip-genotypes", "-i", options.input))
      if (options.split)
        SplitMulti.run(s).vds
      else s.vds
    }

    splitWarning(vds.wasSplit, "VDS", otherVds.wasSplit, "VDS")

    val (finalType, inserter): (Type, (Annotation, Option[Annotation]) => Annotation) = if (expr) {
      val ec = EvalContext(Map(
        "va" -> (0, vds.vaSignature),
        "vds" -> (1, otherVds.vaSignature)))
      buildInserter(code, vds.vaSignature, ec, Annotation.VARIANT_HEAD)
    } else vds.insertVA(otherVds.vaSignature, Parser.parseAnnotationRoot(code, Annotation.VARIANT_HEAD))

    state.copy(vds = vds
      .annotateVariants(otherVds.variantsAndAnnotations, finalType, inserter))
  }
}

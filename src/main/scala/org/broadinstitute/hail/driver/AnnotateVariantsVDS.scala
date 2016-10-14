package org.broadinstitute.hail.driver

import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr.{EvalContext, _}
import org.kohsuke.args4j.{Option => Args4jOption}

object AnnotateVariantsVDS extends Command with JoinAnnotator {

  class Options extends BaseOptions {
    @Args4jOption(name = "-i", aliases = Array("--input"),
      usage = "VDS file path to annotate with")
    var input: String = _

    @Args4jOption(name = "-n", aliases = Array("--name"), usage = "Name of dataset in environment to annotate with")
    var name: String = _

    @Args4jOption(required = false, name = "-r", aliases = Array("--root"),
      usage = "Period-delimited path starting with `va' (this argument or --code required)")
    var root: String = _

    @Args4jOption(required = false, name = "-c", aliases = Array("--code"),
      usage = "Use annotation expressions to join with the table (this argument or --root required)")
    var code: String = _

    @Args4jOption(required = false, name = "--split",
      usage = "split multiallelic variants in the input VDS")
    var split: Boolean = false

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

    if (!((options.input != null) ^ (options.name != null)))
      fatal("either `--input' or `--name' required, but not both")

    var otherVDS =
      if (options.input != null)
        Read.run(state, Array("--skip-genotypes", "-i", options.input)).vds
      else {
        assert(options.name != null)
        state.env.get(options.name) match {
          case Some(vds) => vds
          case None =>
            fatal(s"no such dataset ${ options.name } in environment")
        }
      }

    if (options.split)
      otherVDS = SplitMulti.run(state.copy(vds = otherVDS)).vds

    splitWarning(vds.wasSplit, "VDS", otherVDS.wasSplit, "VDS")

    val (finalType, inserter): (Type, (Annotation, Option[Annotation]) => Annotation) = if (expr) {
      val ec = EvalContext(Map(
        "va" -> (0, vds.vaSignature),
        "vds" -> (1, otherVDS.vaSignature)))
      buildInserter(code, vds.vaSignature, ec, Annotation.VARIANT_HEAD)
    } else vds.insertVA(otherVDS.vaSignature, Parser.parseAnnotationRoot(code, Annotation.VARIANT_HEAD))

    state.copy(vds = vds
      .annotateVariants(otherVDS.variantsAndAnnotations, finalType, inserter))
  }
}

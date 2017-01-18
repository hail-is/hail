package is.hail.driver

import is.hail.utils._
import is.hail.annotations.Annotation
import is.hail.expr.{EvalContext, _}
import is.hail.variant.VariantDataset
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

  }

  def newOptions = new Options

  def name = "annotatevariants vds"

  def description = "Annotate variants with VDS file"

  def supportsMultiallelic = true

  def requiresVDS = true

  def annotate(vds: VariantDataset, other: VariantDataset, code: String, root: String): VariantDataset = {
    if (!((code != null) ^ (root != null)))
      fatal("either `--code' or `--root' required, but not both")

    splitWarning(vds.wasSplit, "VDS", other.wasSplit, "VDS")

    val (finalType, inserter): (Type, (Annotation, Option[Annotation]) => Annotation) = if (code != null) {
      val ec = EvalContext(Map(
        "va" -> (0, vds.vaSignature),
        "vds" -> (1, other.vaSignature)))
      buildInserter(code, vds.vaSignature, ec, Annotation.VARIANT_HEAD)
    } else vds.insertVA(other.vaSignature, Parser.parseAnnotationRoot(root, Annotation.VARIANT_HEAD))

    vds.annotateVariants(other.variantsAndAnnotations, finalType, inserter)
  }

  def run(state: State, options: Options): State = {
    if (!((options.input != null) ^ (options.name != null)))
      fatal("either `--input' or `--name' required, but not both")

    var other =
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

    state.copy(vds = annotate(state.vds, other, options.code, options.root))
  }
}

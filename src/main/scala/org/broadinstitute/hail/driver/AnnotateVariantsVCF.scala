package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.methods.LoadVCF
import org.broadinstitute.hail.expr.{EvalContext, _}
import org.broadinstitute.hail.annotations.Annotation

import scala.collection.JavaConverters._
import org.kohsuke.args4j.{Argument, Option => Args4jOption}

object AnnotateVariantsVCF extends Command with VCFImporter with JoinAnnotator {

  class Options extends BaseOptions {
    @Argument(usage = "<files...>")
    var arguments: java.util.ArrayList[String] = new java.util.ArrayList[String]()

    @Args4jOption(required = false, name = "-r", aliases = Array("--root"),
      usage = "Period-delimited path starting with `va' (this argument or --code required)")
    var root: String = _

    @Args4jOption(required = false, name = "-c", aliases = Array("--code"),
      usage = "Use annotation expressions to join with the table (this argument or --root required)")
    var code: String = _

    @Args4jOption(required = false, name = "--split",
      usage = "split multiallelic variants in VCF")
    var split: Boolean = _

    @Args4jOption(required = false, name = "--force",
      usage = "Force load a .gz file")
    var force: Boolean = _
  }

  def newOptions = new Options

  def name = "annotatevariants vcf"

  def description = "Annotate variants with VCF file"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val (expr, code) = (Option(options.code), Option(options.root)) match {
      case (Some(c), None) => (true, c)
      case (None, Some(r)) => (false, r)
      case _ => fatal("this module requires one of `--root' or `--code', but not both")
    }
    val inputs = globAllVcfs(options.arguments.asScala.toArray, state.hadoopConf, options.force)

    val otherVds = {
      val load = LoadVCF(vds.sparkContext, inputs.head, inputs, skipGenotypes = true)
      if (options.split)
        SplitMulti.run(state.copy(vds = load)).vds
      else load
    }

    splitWarning(vds.wasSplit, "VDS", otherVds.wasSplit, "VCF")

    val (finalType, inserter): (Type, (Annotation, Option[Annotation]) => Annotation) = if (expr) {
      val ec = EvalContext(Map(
        "va" -> (0, vds.vaSignature),
        "vcf" -> (1, otherVds.vaSignature)))
      buildInserter(code, vds.vaSignature, ec, Annotation.VARIANT_HEAD)
    } else vds.insertVA(otherVds.vaSignature, Parser.parseAnnotationRoot(code, Annotation.VARIANT_HEAD))

    state.copy(vds = vds
      .withGenotypeStream()
      .annotateVariants(otherVds.variantsAndAnnotations, finalType, inserter))
  }
}

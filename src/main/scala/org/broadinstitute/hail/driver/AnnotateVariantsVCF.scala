package org.broadinstitute.hail.driver

import org.broadinstitute.hail.methods.LoadVCF
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.annotations.Annotation
import scala.collection.JavaConverters._
import org.kohsuke.args4j.{Argument, Option => Args4jOption}

object AnnotateVariantsVCF extends Command with VCFImporter {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-r", aliases = Array("--root"),
      usage = "Period-delimited path starting with `va'")
    var root: String = _

    @Args4jOption(required = false, name = "--force",
      usage = "Force load a .gz file")
    var force: Boolean = _

    @Argument(usage = "<files...>")
    var arguments: java.util.ArrayList[String] = new java.util.ArrayList[String]()

  }

  def newOptions = new Options

  def name = "annotatevariants vcf"

  def description = "Annotate variants with VCF file"

  def supportsMultiallelic = false

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val inputs = globAllVcfs(options.arguments.asScala.toArray, state.hadoopConf, options.force)

    val otherVds = LoadVCF(vds.sparkContext, inputs.head, inputs, skipGenotypes = true)
    val otherVdsSplit = SplitMulti.run(state.copy(vds = otherVds)).vds

    val annotated = vds
      .withGenotypeStream()
      .annotateVariants(otherVdsSplit.variantsAndAnnotations, otherVdsSplit.vaSignature,
        Parser.parseAnnotationRoot(options.root, Annotation.VARIANT_HEAD))
    state.copy(vds = annotated)
  }
}

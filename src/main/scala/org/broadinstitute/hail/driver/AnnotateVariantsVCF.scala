package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr
import org.broadinstitute.hail.io.annotators._
import org.broadinstitute.hail.methods.LoadVCF
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable

object AnnotateVariantsVCF extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-i", aliases = Array("--input"),
      usage = "VCF file path")
    var input: String = _

    @Args4jOption(required = true, name = "-r", aliases = Array("--root"),
      usage = "Period-delimited path starting with `va'")
    var root: String = _

    @Args4jOption(required = false, name = "--force",
      usage = "Force load a .gz file")
    var force: Boolean = _
  }

  def newOptions = new Options

  def name = "annotatevariants/vcf"

  override def hidden = true

  def description = "Annotate variants in current dataset"

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val filepath = options.input

    fatalIf(!options.force && filepath.endsWith(".gz"), "Hail does not load '.gz' files by default, " +
      "rename to '.bgz' if the file is block-gzipped or use argument '--force'.")

    val otherVds = LoadVCF(vds.sparkContext, filepath).filterSamples({case (s, sa) => false})
    val otherVdsSplit = SplitMulti.run(state.copy(vds = otherVds)).vds

    val annotated = vds.annotateVariants(otherVdsSplit.variantsAndAnnotations, otherVdsSplit.vaSignature,
      AnnotateVariantsTSV.parseRoot(options.root))
    state.copy(vds = annotated)
  }
}

package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.kohsuke.args4j.{Option => Args4jOption}

object AnnotateVariantsVDS extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-i", aliases = Array("--input"),
      usage = "VDS file path")
    var input: String = _

    @Args4jOption(required = true, name = "-r", aliases = Array("--root"),
      usage = "Period-delimited path starting with `va'")
    var root: String = _

  }

  def newOptions = new Options

  def name = "annotatevariants vds"

  def description = "Annotate variants with VDS file"

  def supportsMultiallelic = false

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val filepath = options.input

    val readOtherVds = Read.run(state, Array("--skip-genotypes", "-i", filepath)).vds

    if (!readOtherVds.wasSplit)
      fatal("cannot annotate from a multiallelic VDS, run `splitmulti' on that VDS first.")

    val annotated = vds
      .withGenotypeStream()
      .annotateVariants(readOtherVds.variantsAndAnnotations,
        readOtherVds.vaSignature, Parser.parseAnnotationRoot(options.root, Annotation.VARIANT_HEAD))
    state.copy(vds = annotated)
  }
}

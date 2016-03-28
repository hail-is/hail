package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.io.annotators._
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

  def name = "annotatevariants/vds"

  override def hidden = true

  def description = "Annotate variants in current dataset"

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val filepath = options.input

    val readOtherVds = Read.run(State(state.sc, state.sqlContext, null), Array("-i", filepath)).vds

    fatalIf(!readOtherVds.wasSplit, "cannot annotate from a multiallelic VDS, run 'splitmulti' on that VDS first.")

    val (rdd, signature) =(readOtherVds.variantsAndAnnotations, readOtherVds.vaSignature)
    val annotated = vds.annotateVariants(readOtherVds.variantsAndAnnotations,
      readOtherVds.vaSignature, AnnotateVariantsTSV.parseRoot(options.root))
    state.copy(vds = annotated)
  }
}

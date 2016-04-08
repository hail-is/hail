package org.broadinstitute.hail.driver

object AnnotateVariants extends SuperCommand {
  def name = "annotatevariants"

  def description = "Annotate variants in current dataset"

  override def runCommand(state: State, options: Options): State = throw new UnsupportedOperationException

  override def run(state: State, args: Array[String]): State = {
    val (c, newArgs) = lookup(args)
    c.run(state, newArgs)
  }

  override def run(state: State, options: Options): State = throw new UnsupportedOperationException

  register(AnnotateVariantsBed)
  register(AnnotateVariantsExpr)
  register(AnnotateVariantsIList)
  register(AnnotateVariantsTSV)
  register(AnnotateVariantsVCF)
  register(AnnotateVariantsVDS)
}

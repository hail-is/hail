package org.broadinstitute.hail.driver

object AnnotateVariants extends SuperCommand {
  def name = "annotatevariants"

  def description = "Annotate variants in current dataset"

  override def run(state: State, args: Array[String]): State = {
    val (c, newArgs) = lookup(args)
    c.run(state, newArgs)
  }

  register(AnnotateVariantsBed)
  register(AnnotateVariantsExpr)
  register(AnnotateVariantsIntervals)
  register(AnnotateVariantsJSON)
  register(AnnotateVariantsTable)
  register(AnnotateVariantsVCF)
  register(AnnotateVariantsVDS)
}

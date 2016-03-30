package org.broadinstitute.hail.driver

object AnnotateVariants extends SuperCommand {
  def name = "annotatevariants"

  def description = "Annotate variants in current dataset"

  register(AnnotateVariantsBed)
  register(AnnotateVariantsExpr)
  register(AnnotateVariantsIList)
  register(AnnotateVariantsTSV)
  register(AnnotateVariantsVCF)
  register(AnnotateVariantsVDS)
}

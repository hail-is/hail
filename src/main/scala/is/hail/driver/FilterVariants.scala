package is.hail.driver

object FilterVariants extends SuperCommand {

  def name = "filtervariants"

  def description = "Filter variants in current dataset"

  register(FilterVariantsExpr)
  register(FilterVariantsIntervals)
  register(FilterVariantsList)
  register(FilterVariantsAll)
}

package is.hail.driver

object AnnotateSamples extends SuperCommand {
  def name = "annotatesamples"

  def description = "Annotate samples in current dataset"

  register(AnnotateSamplesExpr)
  register(AnnotateSamplesFam)
  register(AnnotateSamplesTable)
  register(AnnotateSamplesList)
  register(AnnotateSamplesVDS)
}

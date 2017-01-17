package is.hail.driver

object FilterSamples extends SuperCommand {
  def name = "filtersamples"

  def description = "Filter samples in current dataset"

  register(FilterSamplesExpr)
  register(FilterSamplesList)
  register(FilterSamplesAll)
}

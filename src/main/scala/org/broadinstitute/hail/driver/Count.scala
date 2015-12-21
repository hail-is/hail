package org.broadinstitute.hail.driver

object Count extends Command {
  class Options extends BaseOptions
  def newOptions = new Options
  def name = "count"
  def description = "Print number of samples and variants in current dataset"
  def run(state: State, options: Options): State = {
    val vds = state.vds
    println("nSamples = " + vds.nSamples)
    println("nLocalSamples = " + vds.nLocalSamples)
    println("nVariants = " + vds.nVariants)
    state
  }
}

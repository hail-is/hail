package org.broadinstitute.hail.driver

object FilterVariantsAll extends Command {

  class Options extends BaseOptions

  def newOptions = new Options

  def name = "filtervariants all"

  def description = "Discard all variants in the current dataset"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    state.copy(vds = state.vds.copy(rdd = state.sc.emptyRDD))
  }
}

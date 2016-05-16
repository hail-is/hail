package org.broadinstitute.hail.driver

object Cache extends Command {
  class Options extends BaseOptions
  def newOptions = new Options

  def name = "cache"
  def description = "Cache current dataset in memory"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    state.copy(vds = state.vds.cache())
  }
}

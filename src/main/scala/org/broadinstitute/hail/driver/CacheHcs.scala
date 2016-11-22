package org.broadinstitute.hail.driver

object CacheHcs extends Command {
  class Options extends BaseOptions
  def newOptions = new Options

  def name = "cachehcs"
  def description = "Cache current hard call set in memory"
  def run(state: State, options: Options): State = {
    state.copy(hcs = state.hcs.cache())
  }

  def supportsMultiallelic = true

  def requiresVDS = false
}

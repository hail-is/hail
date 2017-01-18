package is.hail.driver

import is.hail.variant.Genotype

object HardCalls extends Command {

  class Options extends BaseOptions

  def newOptions = new Options

  def name = "hardcalls"

  def description = "Drop all genotype fields except the GT field"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    state.copy(
      vds = state.vds.mapValues { g =>
        Genotype(g.gt, g.fakeRef)
      })
  }
}

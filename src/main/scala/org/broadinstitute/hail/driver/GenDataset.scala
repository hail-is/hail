package org.broadinstitute.hail.driver

import org.broadinstitute.hail.variant.{Genotype, VDSGens, VariantSampleMatrix}

object GenDataset extends Command {

  class Options extends BaseOptions

  def newOptions = new Options

  def name = "gen"

  def description = "Generate random dataset"

  override def hidden = true

  def supportsMultiallelic = true

  def requiresVDS = false

  def run(state: State, options: Options): State = {
    state.copy(
      vds = VariantSampleMatrix.gen[Genotype](state.sc, VDSGens.realistic).sample())
  }
}

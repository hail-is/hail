package org.broadinstitute.hail.driver

import org.broadinstitute.hail.variant.{Genotype, VariantSampleMatrix}

object GenDataset extends ImportCommand {

  class Options extends BaseOptions

  def newOptions = new Options

  def name = "gendataset"

  def description = "Generate random dataset"

  override def hidden = true

  def run(state: State, options: Options): State = {
    state.copy(
      vds = VariantSampleMatrix.gen(state.sc, Genotype.gen).sample())
  }
}

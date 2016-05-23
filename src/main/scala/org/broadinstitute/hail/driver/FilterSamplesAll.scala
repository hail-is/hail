package org.broadinstitute.hail.driver

import org.kohsuke.args4j.{Option => Args4jOption}

object FilterSamplesAll extends Command {

  class Options extends BaseOptions

  def newOptions = new Options

  def name = "filtersamples all"

  def description = "Discard all samples in current dataset"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    state.copy(vds = state.vds.dropSamples())
  }
}

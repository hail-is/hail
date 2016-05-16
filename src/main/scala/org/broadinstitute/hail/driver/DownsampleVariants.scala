package org.broadinstitute.hail.driver

import org.kohsuke.args4j.{Option => Args4jOption}

object DownsampleVariants extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "--keep", usage = "(Expected) number of variants to keep")
    var keep: Long = _
  }

  def newOptions = new Options

  def name = "downsamplevariants"

  def description = "Downsample variants in current dataset"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val nVariants = state.vds.nVariants
    val vds = state.vds
    state.copy(vds = vds.sampleVariants(options.keep.toDouble / nVariants))
  }
}

package org.broadinstitute.hail.driver

import org.broadinstitute.hail.utils._
import org.kohsuke.args4j.{Option => Args4jOption}


object Filtermulti extends Command {

  def name = "filtermulti"

  def description = "Filter multi-allelic sites in the current dataset.  " +
    "Useful for running commands that require biallelic variants without an expensive `splitmulti' step."

  class Options extends BaseOptions {
  }

  def newOptions = new Options

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {

    val vds = state.vds
    if (vds.wasSplit) {
      warn("called redundant `filtermulti' on an already split or multiallelic-filtered VDS")
      return state
    }

    val newVDS = vds.filterVariants({
      case (v,va,gs) => v.isBiallelic
    })

    state.copy(
      vds = newVDS.copy(wasSplit = true)
    )

  }
}

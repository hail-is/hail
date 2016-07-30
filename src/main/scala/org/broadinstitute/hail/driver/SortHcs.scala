package org.broadinstitute.hail.driver

import org.kohsuke.args4j.{Option => Args4jOption}

object SortHcs extends Command {
  class Options extends BaseOptions

  def newOptions = new Options

  def name = "sorthcs"
  def description = "Sort the current hard call set by variant: chrom, pos, ref, alt"

  def run(state: State, options: Options): State = {
    state.copy(hcs = state.hcs.sortByVariant())
  }

  def supportsMultiallelic = true

  def requiresVDS = false
}

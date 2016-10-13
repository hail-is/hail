package org.broadinstitute.hail.driver

import org.broadinstitute.hail.utils._
import org.kohsuke.args4j.{Option => Args4jOption}

object CompareVDS extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-n", aliases = Array("--name"), usage = "Name of dataset in environment to compare to")
    var name: String = _
  }

  def newOptions = new Options

  def name = "comparevds"

  def description = "Print number of samples, variants, and called genotypes in current dataset"

  def supportsMultiallelic = true

  def requiresVDS = true

  override def hidden = true

  def run(state: State, options: Options): State = {
    val vds1 = state.vds
    val vds2 = state.env.get(options.name) match {
      case Some(vds) => vds
      case None =>
        fatal("no such dataset $name in environment")
    }

    println(vds1.same(vds2))

    state
  }
}

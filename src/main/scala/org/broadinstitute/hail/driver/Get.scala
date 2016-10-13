package org.broadinstitute.hail.driver

import org.broadinstitute.hail.utils._
import org.kohsuke.args4j.{Option => Args4jOption}

object Get extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-n", aliases = Array("--name"),
      usage = "Name of dataset to get")
    var name: String = _
  }

  def newOptions = new Options

  def name = "get"

  def description = "Get a dataset from environment"

  def supportsMultiallelic = true

  def requiresVDS = false

  def run(state: State, options: Options): State = {
    val name = options.name
    state.env.get(options.name) match {
      case Some(newVDS) =>
        state.copy(vds = newVDS)
      case None =>
        fatal("no such dataset $name in environment")
    }
  }
}

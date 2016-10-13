package org.broadinstitute.hail.driver

import org.kohsuke.args4j.{Option => Args4jOption}

object Put extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-n", aliases = Array("--name"),
      usage = "Name of dataset to put")
    var name: String = _
  }

  def newOptions = new Options

  def name = "put"

  def description = "Put a dataset in the environment"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val name = options.name
    state.copy(
      env = state.env + (name -> state.vds))
  }
}

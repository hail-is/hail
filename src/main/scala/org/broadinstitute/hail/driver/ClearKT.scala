package org.broadinstitute.hail.driver

import org.kohsuke.args4j.{Option => Args4jOption}

object ClearKT extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-n", aliases = Array("--name"),
      usage = "Name of key table to clear")
    var name: String = _
  }

  def newOptions = new Options

  def name = "ktclear"

  def description = "Clear key table from environment"

  def supportsMultiallelic = true

  def requiresVDS = false

  def run(state: State, options: Options): State = {
    val name = options.name
    state.copy(
      ktEnv = state.ktEnv - name)
  }
}

package org.broadinstitute.hail.driver.keytable

import org.broadinstitute.hail.driver.{Command, State}
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

  override def hidden = true

  def run(state: State, options: Options): State = {
    val name = options.name
    // Fixme: check name in state first
    state.copy(
      ktEnv = state.ktEnv - name)
  }
}

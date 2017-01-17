package is.hail.driver

import org.kohsuke.args4j.{Option => Args4jOption}

object Clear extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-n", aliases = Array("--name"),
      usage = "Name of dataset to clear")
    var name: String = _
  }

  def newOptions = new Options

  def name = "clear"

  def description = "Clear dataset from environment"

  def supportsMultiallelic = true

  def requiresVDS = false

  def run(state: State, options: Options): State = {
    val name = options.name
    state.copy(
      env = state.env - name)
  }
}

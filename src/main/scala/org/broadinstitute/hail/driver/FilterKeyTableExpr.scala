package org.broadinstitute.hail.driver

import org.broadinstitute.hail.utils._
import org.kohsuke.args4j.{Option => Args4jOption}

object FilterKeyTableExpr extends Command {
  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-c", aliases = Array("--cond"),
    usage = "Boolean expression for filtering", metaVar = "EXPR")
    var condition: String = _

    @Args4jOption(required = true, name = "-n", aliases = Array("--name"),
      usage = "Name of source key table")
    var name: String = _

    @Args4jOption(required = false, name = "-d", aliases = Array("--dest"),
      usage = "Name of destination key table (can be same as source)")
    var dest: String = _

    @Args4jOption(required = false, name = "--keep", usage = "Keep variants matching condition")
    var keep: Boolean = false

    @Args4jOption(required = false, name = "--remove", usage = "Remove variants matching condition")
    var remove: Boolean = false
  }

  def newOptions = new Options

  def name = "filterkeytable expr"

  def description = "Filter key table using a boolean expression"

  def supportsMultiallelic = true

  def requiresVDS = false

  override def hidden = true

  def run(state: State, options: Options): State = {
    val kt = state.ktEnv.get(options.name) match {
      case Some(newKT) =>
        newKT
      case None =>
        fatal("no such key table $name in environment")
    }

    if (!(options.keep ^ options.remove))
      fatal("either `--keep' or `--remove' required, but not both")

    val cond = options.condition
    val keep = options.keep
    val dest = if (options.dest != null) options.dest else options.name

    state.copy(ktEnv = state.ktEnv + ( dest -> kt.filterExpr(cond, keep)))
  }
}

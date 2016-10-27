package org.broadinstitute.hail.driver.keytable

import org.broadinstitute.hail.driver.{Command, State}
import org.broadinstitute.hail.utils._
import org.kohsuke.args4j.{Option => Args4jOption}

object WriteKeyTable extends Command {
  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-n", aliases = Array("--name"),
      usage = "Name of key table")
    var name: String = _

    @Args4jOption(required = true, name = "-o", aliases = Array("--output"),
      usage = "Output file path")
    var output: String = _
  }

  def newOptions = new Options

  def name = "ktwrite"

  def description = "Write key table to disk"

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

    kt.write(state.sqlContext, options.output)

    state
  }
}

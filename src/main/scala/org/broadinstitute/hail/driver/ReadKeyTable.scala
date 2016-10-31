package org.broadinstitute.hail.driver

import org.broadinstitute.hail.keytable.KeyTable
import org.broadinstitute.hail.utils._
import org.kohsuke.args4j.{Option => Args4jOption}

object ReadKeyTable extends Command {
  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-n", aliases = Array("--name"),
      usage = "Name of key table")
    var name: String = _

    @Args4jOption(required = true, name = "-i", aliases = Array("--input"),
      usage = "Input file path")
    var input: String = _
  }

  def newOptions = new Options

  def name = "ktread"

  def description = "Load key table from disk"

  def supportsMultiallelic = true

  def requiresVDS = false

  override def hidden = true

  def run(state: State, options: Options): State = {

    if (state.ktEnv.contains(options.name))
      fatal("key table $name already exists in environment")

    val kt: KeyTable = KeyTable.read(state.sqlContext, options.input)

    state.copy(ktEnv = state.ktEnv + (options.name -> kt))
  }
}


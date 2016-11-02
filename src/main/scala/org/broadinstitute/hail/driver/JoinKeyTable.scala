package org.broadinstitute.hail.driver

import org.apache.spark.sql.Row
import org.broadinstitute.hail.expr.{EvalContext, _}
import org.broadinstitute.hail.io.TextExporter
import org.broadinstitute.hail.utils._
import org.kohsuke.args4j.{Option => Args4jOption}

object JoinKeyTable extends Command {

  class Options extends BaseOptions {

    @Args4jOption(required = true, name = "-d", aliases = Array("--dest"),
      usage = "name of joined key-table")
    var destName: String = _

    @Args4jOption(required = true, name = "-l", aliases = Array("--left-name"),
      usage = "name of key-table on left")
    var leftName: String = _

    @Args4jOption(required = true, name = "-r", aliases = Array("--right-name"),
      usage = "name of key-table on right")
    var rightName: String = _

    @Args4jOption(required = false, name = "-t", aliases = Array("--join-type"),
      usage = "type of join")
    var joinType: String = "left"

    @Args4jOption(required = true, name = "-t", aliases = Array("--join-keys"),
      usage = "name of columns to join on")
    var joinKeys: String = _
  }

  def newOptions = new Options

  def name = "joinkeytable"

  def description = "Join two key tables together to produce new key table"

  def supportsMultiallelic = true

  def requiresVDS = false

  override def hidden = true

  def run(state: State, options: Options): State = {
    val ktEnv = state.ktEnv

    val ktLeft = ktEnv.get(options.leftName) match {
      case Some(kt) =>
        kt
      case None =>
        fatal("no such key table $name in environment")
    }

    val ktRight = ktEnv.get(options.rightName) match {
      case Some(kt) =>
        kt
      case None =>
        fatal("no such key table $name in environment")
    }

    if (ktEnv.contains(options.destName))
      warn("destination name already exists -- overwriting previous key-table")




    state
  }
}


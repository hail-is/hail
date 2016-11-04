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

    @Args4jOption(required = true, name = "-k", aliases = Array("--join-keys"),
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
    val leftName = options.leftName
    val rightName = options.rightName
    val dest = options.destName

    val ktLeft = ktEnv.get(leftName) match {
      case Some(kt) => kt
      case None => fatal("no such key table $leftName in environment")
    }

    val ktRight = ktEnv.get(rightName) match {
      case Some(kt) => kt
      case None => fatal("no such key table $rightName in environment")
    }

    if (ktEnv.contains(dest))
      warn("destination name already exists -- overwriting previous key-table")

    val ktLeftFieldSet = ktLeft.fieldNames.toSet
    val ktRightFieldSet = ktRight.fieldNames.toSet

    val joinKeys = if (options.joinKeys == null) ktLeft.keyNames.toArray else Parser.parseIdentifierList(options.joinKeys)

    if (!joinKeys.forall(k => ktLeftFieldSet.contains(k)) || !joinKeys.forall(k => ktRightFieldSet.contains(k)))
      fatal(
        s"""Join keys not present in both key-tables.
            |Keys found: ${ joinKeys.mkString(",") }
            |Left KeyTable Schema: ${ ktLeft.schema }
            |Right KeyTable Schema: ${ ktRight.schema }
         """.stripMargin)

    val joinedKT = options.joinType match {
      case "left" => ktLeft.leftJoin(ktRight, joinKeys)
      case "right" => ktLeft.rightJoin(ktRight, joinKeys)
      case "inner" => ktLeft.innerJoin(ktRight, joinKeys)
      case "outer" => ktLeft.outerJoin(ktRight, joinKeys)
      case _ => fatal("Did not recognize join type. Pick one of [left, right, inner, outer].")
    }

    state.copy(ktEnv = state.ktEnv + (dest -> joinedKT))
  }
}


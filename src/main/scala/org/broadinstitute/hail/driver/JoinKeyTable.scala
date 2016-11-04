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

    if (ktLeft.keySignature != ktRight.keySignature)
      fatal(
        s"""Key schemas are not Identical.
            |Left KeyTable Schema: ${ ktLeft.keySchema }
            |Right KeyTable Schema: ${ ktRight.keySchema }
         """.stripMargin)

    val valueDuplicates = ktLeft.valueNames.intersect(ktRight.valueNames)
    if (valueDuplicates.nonEmpty)
      fatal(
        s"""Invalid join operation: cannot merge key-tables with same-name fields.
            |Found these fields in both tables: [ ${ valueDuplicates.mkString(", ") } ]
         """.stripMargin)

    val joinedKT = options.joinType match {
      case "left" => ktLeft.leftJoin(ktRight)
      case "right" => ktLeft.rightJoin(ktRight)
      case "inner" => ktLeft.innerJoin(ktRight)
      case "outer" => ktLeft.outerJoin(ktRight)
      case _ => fatal("Did not recognize join type. Pick one of [left, right, inner, outer].")
    }

    state.copy(ktEnv = state.ktEnv + (dest -> joinedKT))
  }
}


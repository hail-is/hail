package is.hail.expr.types

import is.hail.expr.Parser
import is.hail.expr.ir._
import is.hail.rvd.OrderedRVDType
import is.hail.utils._
import org.json4s.CustomSerializer
import org.json4s.JsonAST.JString

class TableTypeSerializer extends CustomSerializer[TableType](format => (
  { case JString(s) => Parser.parseTableType(s) },
  { case tt: TableType => JString(tt.toString) }))

case class TableType(rowType: TStruct, key: Option[IndexedSeq[String]], globalType: TStruct) extends BaseType {
  assert(!key.exists(_.isEmpty))

  val keyOrEmpty: IndexedSeq[String] = key.getOrElse(IndexedSeq.empty)
  val keyOrNull: IndexedSeq[String] = key.orNull
  val rvdType = OrderedRVDType(keyOrEmpty, rowType)

  def env: Env[Type] = {
    Env.empty[Type]
      .bind(("global", globalType))
      .bind(("row", rowType))
  }

  def globalEnv: Env[Type] = Env.empty[Type]
    .bind("global" -> globalType)

  def rowEnv: Env[Type] = Env.empty[Type]
    .bind("global" -> globalType)
    .bind("row" -> rowType)

  def refMap: Map[String, Type] = Map(
    "global" -> globalType,
    "row" -> rowType)

  def keyType: Option[TStruct] = key.map(_ => rvdType.kType)
  val keyFieldIdx: Option[Array[Int]] = key.map(_ => rvdType.kFieldIdx)
  def valueType: TStruct = rvdType.valueType
  val valueFieldIdx: Array[Int] = rvdType.valueFieldIdx

  def pretty(sb: StringBuilder, indent0: Int = 0, compact: Boolean = false) {
    var indent = indent0

    val space: String = if (compact) "" else " "

    def newline() {
      if (!compact) {
        sb += '\n'
        sb.append(" " * indent)
      }
    }

    sb.append(s"Table$space{")
    indent += 4
    newline()

    sb.append(s"global:$space")
    globalType.pretty(sb, indent, compact)
    sb += ','
    newline()

    key match {
      case Some(key) =>
        sb.append(s"key:$space[")
        key.foreachBetween(k => sb.append(prettyIdentifier(k)))(sb.append(s",$space"))
        sb += ']'
      case None =>
        sb.append(s"key:${space}None")
    }
    sb += ','
    newline()

    sb.append(s"row:$space")
    rowType.pretty(sb, indent, compact)

    indent -= 4
    newline()
    sb += '}'
  }
}

package is.hail.expr.types

import is.hail.expr.Parser
import is.hail.expr.ir._
import is.hail.rvd.RVDType
import is.hail.utils._
import org.json4s.CustomSerializer
import org.json4s.JsonAST.JString

class TableTypeSerializer extends CustomSerializer[TableType](format => (
  { case JString(s) => Parser.parseTableType(s) },
  { case tt: TableType => JString(tt.toString) }))

case class TableType(rowType: TStruct, key: IndexedSeq[String], globalType: TStruct) extends BaseType {
  val rvdType = RVDType(rowType.physicalType, key)

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

  def keyType: TStruct = rvdType.kType.virtualType
  val keyFieldIdx: Array[Int] = rvdType.kFieldIdx
  def valueType: TStruct = rvdType.valueType.virtualType
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

    sb.append(s"key:$space[")
    key.foreachBetween(k => sb.append(prettyIdentifier(k)))(sb.append(s",$space"))
    sb += ']'
    sb += ','
    newline()

    sb.append(s"row:$space")
    rowType.pretty(sb, indent, compact)

    indent -= 4
    newline()
    sb += '}'
  }
}

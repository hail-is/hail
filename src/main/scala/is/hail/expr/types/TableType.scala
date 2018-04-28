package is.hail.expr.types

import is.hail.expr.{EvalContext, Parser}
import is.hail.utils._
import is.hail.expr.ir._
import org.json4s.CustomSerializer
import org.json4s.JsonAST.JString

class TableTypeSerializer extends CustomSerializer[TableType](format => (
  { case JString(s) => Parser.parseTableType(s) },
  { case tt: TableType => JString(tt.toString) }))

case class TableType(rowType: TStruct, key: IndexedSeq[String], globalType: TStruct) extends BaseType {
  def env: Env[Type] = {
    Env.empty[Type]
      .bind(("global", globalType))
      .bind(("row", rowType))
  }

  def tAgg: TAggregable = TAggregable(rowType, Map(
    "global" -> (0, globalType),
    "row" -> (1, rowType)
  ))

  def aggEnv: Env[Type] = Env.empty[Type]
    .bind("global" -> globalType)
    .bind("AGG" -> tAgg)

  def keyType: TStruct = rowType.select(key.toArray)._1
  val keyFieldIdx: Array[Int] = key.toArray.map(rowType.fieldIdx)
  def valueType: TStruct = rowType.filter(key.toSet, include = false)._1
  val valueFieldIdx: Array[Int] =
    rowType.fields.filter(f => !key.contains(f.name)).map(_.index).toArray

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

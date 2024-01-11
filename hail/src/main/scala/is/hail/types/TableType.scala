package is.hail.types

import is.hail.expr.ir._
import is.hail.rvd.RVDType
import is.hail.types.physical.{PStruct, PType}
import is.hail.types.virtual.{TStruct, Type}
import is.hail.utils._

import org.json4s.{CustomSerializer, _}
import org.json4s.JsonAST.JString

class TableTypeSerializer extends CustomSerializer[TableType](format =>
      (
        { case JString(s) => IRParser.parseTableType(s) },
        { case tt: TableType => JString(tt.toString) },
      )
    )

object TableType {
  def keyType(ts: TStruct, key: IndexedSeq[String]): TStruct =
    ts.typeAfterSelect(key.map(ts.fieldIdx))

  def valueType(ts: TStruct, key: IndexedSeq[String]): TStruct =
    ts.filterSet(key.toSet, include = false)._1

  val minimal: TableType =
    TableType(
      TStruct.empty,
      FastSeq(),
      TStruct.empty,
    )
}

case class TableType(rowType: TStruct, key: IndexedSeq[String], globalType: TStruct)
    extends BaseType {
  lazy val canonicalRowPType = PType.canonical(rowType).setRequired(true).asInstanceOf[PStruct]
  lazy val canonicalRVDType = RVDType(canonicalRowPType, key)

  key.foreach { k =>
    if (!rowType.hasField(k))
      throw new RuntimeException(s"key field $k not in row type: $rowType")
  }

  @transient lazy val globalEnv: Env[Type] = Env.empty[Type]
    .bind("global" -> globalType)

  @transient lazy val rowEnv: Env[Type] = Env.empty[Type]
    .bind("global" -> globalType)
    .bind("row" -> rowType)

  @transient lazy val refMap: Map[String, Type] = Map(
    "global" -> globalType,
    "row" -> rowType,
  )

  def isCanonical: Boolean = rowType.isCanonical && globalType.isCanonical

  lazy val keyType: TStruct = TableType.keyType(rowType, key)
  def keyFieldIdx: Array[Int] = canonicalRVDType.kFieldIdx
  lazy val valueType: TStruct = TableType.valueType(rowType, key)
  def valueFieldIdx: Array[Int] = canonicalRVDType.valueFieldIdx

  def pretty(sb: StringBuilder, indent0: Int = 0, compact: Boolean = false): Unit = {
    var indent = indent0

    val space: String = if (compact) "" else " "

    def newline(): Unit =
      if (!compact) {
        sb += '\n'
        sb.append(" " * indent)
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

  def toJSON: JObject =
    JObject(
      "global_type" -> JString(globalType.toString),
      "row_type" -> JString(rowType.toString),
      "row_key" -> JArray(key.map(f => JString(f)).toList),
    )
}

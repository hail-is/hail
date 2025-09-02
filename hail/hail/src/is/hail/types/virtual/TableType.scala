package is.hail.types.virtual

import is.hail.expr.ir._
import is.hail.rvd.RVDType
import is.hail.types.physical.{PStruct, PType}
import is.hail.utils._

import org.json4s._
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

  def globalBindings: IndexedSeq[Int] = FastSeq(0)

  def rowBindings: IndexedSeq[Int] = FastSeq(0, 1)
}

case class TableType(rowType: TStruct, key: IndexedSeq[String], globalType: TStruct) extends VType {
  lazy val canonicalRowPType = PType.canonical(rowType).setRequired(true).asInstanceOf[PStruct]
  lazy val canonicalRVDType = RVDType(canonicalRowPType, key)

  key.foreach { k =>
    if (!rowType.hasField(k))
      throw new RuntimeException(s"key field $k not in row type: $rowType")
  }

  @transient lazy val globalEnv: Env[Type] =
    Env.empty[Type].bind(globalBindings: _*)

  def globalBindings: IndexedSeq[(Name, Type)] =
    FastSeq(TableIR.globalName -> globalType)

  @transient lazy val rowEnv: Env[Type] =
    Env.empty[Type].bind(rowBindings: _*)

  def rowBindings: IndexedSeq[(Name, Type)] =
    FastSeq(TableIR.globalName -> globalType, TableIR.rowName -> rowType)

  def isCanonical: Boolean = rowType.isCanonical && globalType.isCanonical

  lazy val keyType: TStruct = TableType.keyType(rowType, key)
  def keyFieldIdx: Array[Int] = canonicalRVDType.kFieldIdx
  lazy val valueType: TStruct = TableType.valueType(rowType, key)
  def valueFieldIdx: Array[Int] = canonicalRVDType.valueFieldIdx

  def pretty(sb: StringBuilder, indent0: Int = 0, compact: Boolean = false): Unit = {

    val space: String = if (compact) "" else " "
    val padding: String = if (compact) "" else " " * indent0
    val newline: String = if (compact) "" else "\n"

    val indent = indent0 + 4

    sb ++= "Table" ++= space += '{' ++= newline: Unit

    sb ++= padding ++= "global:" ++= space: Unit
    globalType.pretty(sb, indent, compact)
    sb += ',' ++= newline: Unit

    sb ++= padding ++= "key:" ++= space += '[': Unit
    key.foreachBetween(k => sb ++= prettyIdentifier(k))(sb += ',' ++= space: Unit)
    sb ++= "]," ++ newline

    sb ++= padding ++= "row:" ++= space: Unit
    rowType.pretty(sb, indent, compact)

    sb ++= newline += '}': Unit
  }

  override def toJSON: JObject =
    JObject(
      "global_type" -> JString(globalType.toString),
      "row_type" -> JString(rowType.toString),
      "row_key" -> JArray(key.map(f => JString(f)).toList),
    )
}

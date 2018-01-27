package is.hail.expr.types

import is.hail.expr.EvalContext
import is.hail.utils._
import is.hail.expr.ir._

case class TableType(rowType: TStruct, key: Array[String], globalType: TStruct) extends BaseType {
  def rowEC: EvalContext = EvalContext(rowType.fields.map { f => f.name -> f.typ } ++
      globalType.fields.map { f => f.name -> f.typ }: _*)
  def fields: Map[String, Type] = Map(rowType.fields.map { f => f.name -> f.typ } ++ globalType.fields.map { f => f.name -> f.typ }: _*)

  def env: Env[Type] = {
    Env.empty[Type]
      .bind(rowType.fields.map {f => (f.name, f.typ) }:_*)
      .bind(globalType.fields.map {f => (f.name, f.typ) }:_*)
  }
  def remapIR(ir: IR): IR = ir match {
    case Ref(y, _) if rowType.selfField(y).isDefined => GetField(In(0, rowType), y)
    case Ref(y, _) if globalType.selfField(y).isDefined => GetField(In(1, globalType), y)
    case ir2 => Recur(remapIR)(ir2)
  }

  def pretty(sb: StringBuilder, indent: Int = 0, compact: Boolean = false) {
    // FIXME compact
    sb.append("Table{")

    sb.append("global:")
    globalType.pretty(sb, indent, compact)
    sb += ','

    sb.append("key:[")
    key.foreachBetween(k => sb.append(prettyIdentifier(k)))(sb += ',')
    sb += ']'
    sb += ','

    sb.append("row:")
    rowType.pretty(sb, indent, compact)
    sb += '}'
  }
}

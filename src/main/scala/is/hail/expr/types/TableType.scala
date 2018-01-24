package is.hail.expr.types

import is.hail.expr.EvalContext
import is.hail.expr.ir._

/**
  * Created by cotton on 1/24/18.
  */
case class TableType(rowType: TStruct, key: Array[String], globalType: TStruct) extends BaseType {
  def rowEC: EvalContext = EvalContext(rowType.fields.map { f => f.name -> f.typ } ++
      globalType.fields.map { f => f.name -> f.typ }: _*)
  def fields: Map[String, Type] = Map(rowType.fields.map { f => f.name -> f.typ } ++ globalType.fields.map { f => f.name -> f.typ }: _*)

  def env: Env[Type] = {
    new Env[Type]()
      .bind(rowType.fields.map {f => (f.name, f.typ) }:_*)
      .bind(globalType.fields.map {f => (f.name, f.typ) }:_*)
  }
  def remapIR(ir: IR): IR = ir match {
    case Ref(y, _) if rowType.selfField(y).isDefined => GetField(In(0, rowType), y)
    case Ref(y, _) if globalType.selfField(y).isDefined => GetField(In(1, globalType), y)
    case ir2 => Recur(remapIR)(ir2)
  }
}

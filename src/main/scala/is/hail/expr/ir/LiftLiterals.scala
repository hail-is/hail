package is.hail.expr.ir

import is.hail.HailContext
import is.hail.annotations.BroadcastRow
import is.hail.expr.types.{TStruct, Type}
import is.hail.utils.ArrayBuilder
import org.apache.spark.sql.Row

import scala.collection.mutable

object LiftLiterals {
  lazy val emptyRow: BroadcastRow = BroadcastRow.empty(HailContext.get.sc)

  def getLiterals(irs: IR*): BroadcastRow = {
    val included = mutable.Set.empty[String]
    val ab = new ArrayBuilder[Literal]()

    def visit(ir: IR): Unit = {
      ir match {
        case l: Literal =>
          if (!included.contains(l.id)) {
            ab += l
            included += l.id
          }
        case _ =>
          ir.children.foreach {
            case ir: IR => visit(ir)
            case _ =>
          }
      }
    }

    irs.foreach(visit)
    val literals = ab.result()

    BroadcastRow(
      Row.fromSeq(literals.map(_.value)),
      TStruct(literals.map(l => l.id -> l.typ): _*),
      HailContext.get.sc)
  }

  def addLiterals(tir: TableIR, literals: BroadcastRow): TableIR = {
    TableMapGlobals(tir,
      InsertFields(
        Ref("global", tir.typ.globalType),
        literals.t.fields.map(f => f.name -> GetField(Ref("value", literals.t), f.name))),
      literals)
  }

  def addLiterals(mir: MatrixIR, literals: BroadcastRow): MatrixIR = {
    MatrixMapGlobals(mir,
      InsertFields(
        Ref("global", mir.typ.globalType),
        literals.t.fields.map(f => f.name -> GetField(Ref("value", literals.t), f.name))),
      literals)
  }

  def removeLiterals(tir: TableIR, literals: BroadcastRow): TableIR = {
    val literalFields = literals.t.fields.map(_.name).toSet
    TableMapGlobals(tir,
      SelectFields(
        Ref("global", tir.typ.globalType),
        tir.typ.globalType.fieldNames.filter(f => !literalFields.contains(f))),
      emptyRow)
  }

  def removeLiterals(mir: MatrixIR, literals: BroadcastRow): MatrixIR = {
    val literalFields = literals.t.fields.map(_.name).toSet
    MatrixMapGlobals(mir,
      SelectFields(
        Ref("global", mir.typ.globalType),
        mir.typ.globalType.fieldNames.filter(f => !literalFields.contains(f))),
      emptyRow)
  }

  def rewriteIR(ir: IR, newGlobalType: Type): IR = {
    MapIR.mapBaseIR(ir, {
      case Ref("global", _) => Ref("global", newGlobalType)
      case Literal(_, _, id) => GetField(Ref("global", newGlobalType), id)
      case ir => ir
    }).asInstanceOf[IR]
  }

  def apply(ir: BaseIR): BaseIR = {
    MapIR.mapBaseIR(ir, {
      case MatrixMapRows(child, newRow, newKey) =>
        val literals = getLiterals(newRow)
        val rewriteChild = addLiterals(child, literals)
        removeLiterals(
          MatrixMapRows(rewriteChild, rewriteIR(newRow, rewriteChild.typ.globalType), newKey),
          literals)
      case MatrixMapEntries(child, newEntries) =>
        val literals = getLiterals(newEntries)
        val rewriteChild = addLiterals(child, literals)
        removeLiterals(
          MatrixMapEntries(rewriteChild, rewriteIR(newEntries, rewriteChild.typ.globalType)),
          literals)
      case MatrixMapCols(child, newRow, newKey) =>
        val literals = getLiterals(newRow)
        val rewriteChild = addLiterals(child, literals)
        removeLiterals(
          MatrixMapCols(rewriteChild, rewriteIR(newRow, rewriteChild.typ.globalType), newKey),
          literals)
      case MatrixFilterRows(child, pred) =>
        val literals = getLiterals(pred)
        val rewriteChild = addLiterals(child, literals)
        removeLiterals(
          MatrixFilterRows(rewriteChild, rewriteIR(pred, rewriteChild.typ.globalType)),
          literals)
      case MatrixFilterCols(child, pred) =>
        val literals = getLiterals(pred)
        val rewriteChild = addLiterals(child, literals)
        removeLiterals(
          MatrixFilterCols(rewriteChild, rewriteIR(pred, rewriteChild.typ.globalType)),
          literals)
      case MatrixFilterEntries(child, pred) =>
        val literals = getLiterals(pred)
        val rewriteChild = addLiterals(child, literals)
        removeLiterals(
          MatrixFilterEntries(rewriteChild, rewriteIR(pred, rewriteChild.typ.globalType)),
          literals)
      case MatrixAggregateRowsByKey(child, aggIR) =>
        val literals = getLiterals(aggIR)
        val rewriteChild = addLiterals(child, literals)
        removeLiterals(
          MatrixAggregateRowsByKey(rewriteChild, rewriteIR(aggIR, rewriteChild.typ.globalType)),
          literals)
      case MatrixAggregateColsByKey(child, aggIR) =>
        val literals = getLiterals(aggIR)
        val rewriteChild = addLiterals(child, literals)
        removeLiterals(
          MatrixAggregateColsByKey(rewriteChild, rewriteIR(aggIR, rewriteChild.typ.globalType)),
          literals)
      case TableFilter(child, pred) =>
        val literals = getLiterals(pred)
        val rewriteChild = addLiterals(child, literals)
        removeLiterals(
          TableFilter(rewriteChild, rewriteIR(pred, rewriteChild.typ.globalType)),
          literals)
      case TableMapRows(child, newRow, newKey, preservedKey) =>
        val literals = getLiterals(newRow)
        val rewriteChild = addLiterals(child, literals)
        removeLiterals(
          TableMapRows(rewriteChild, rewriteIR(newRow, rewriteChild.typ.globalType), newKey, preservedKey),
          literals)
      case TableKeyByAndAggregate(child, expr, newKey, nPartitions, bufferSize) =>
        val literals = getLiterals(expr, newKey)
        val rewriteChild = addLiterals(child, literals)
        removeLiterals(
          TableKeyByAndAggregate(rewriteChild, rewriteIR(expr, rewriteChild.typ.globalType),
            rewriteIR(newKey, rewriteChild.typ.globalType), nPartitions, bufferSize),
          literals)
      case TableAggregateByKey(child, expr) =>
        val literals = getLiterals(expr)
        val rewriteChild = addLiterals(child, literals)
        removeLiterals(
          TableAggregateByKey(rewriteChild, rewriteIR(expr, rewriteChild.typ.globalType)),
          literals)
      case TableAggregate(child, expr) =>
        val literals = getLiterals(expr)
        val rewriteChild = addLiterals(child, literals)
        TableAggregate(rewriteChild, rewriteIR(expr, rewriteChild.typ.globalType))
      case MatrixAggregate(child, expr) =>
        val literals = getLiterals(expr)
        val rewriteChild = addLiterals(child, literals)
        MatrixAggregate(rewriteChild, rewriteIR(expr, rewriteChild.typ.globalType))
      case ir => ir
    })
  }
}

package is.hail.expr.ir

import is.hail.annotations.BroadcastRow
import is.hail.expr.types.virtual.{TStruct, Type}
import is.hail.utils._

import scala.collection.mutable

object LiftNonCompilable {
  def extractNonCompilable(irs: IR*): Map[String, IR] = {
    val included = mutable.Set.empty[IR]

    def visit(ir: IR): Unit = {
      if (!Compilable(ir) && !included.contains(ir))
        included += ir
      ir.children.foreach {
        case ir: IR => visit(ir)
        case _ =>
      }
    }

    irs.foreach(visit)
    included.toArray.map { l => genUID() -> l }.toMap
  }

  def addLiterals(tir: TableIR, literals: Map[String, IR]): TableIR = {
    if (literals.isEmpty)
      tir
    else
      TableMapGlobals(tir,
        InsertFields(
          Ref("global", tir.typ.globalType),
          literals.toFastIndexedSeq))
  }

  def addLiterals(mir: MatrixIR, literals: Map[String, IR]): MatrixIR = {
    if (literals.isEmpty)
      mir
    else
      MatrixMapGlobals(mir,
        InsertFields(
          Ref("global", mir.typ.globalType),
          literals.toFastIndexedSeq))
  }

  def removeLiterals(tir: TableIR, literals: Map[String, IR]): TableIR = {
    if (literals.isEmpty)
      tir
    else {
      val literalFields = literals.keySet
      TableMapGlobals(tir,
        SelectFields(
          Ref("global", tir.typ.globalType),
          tir.typ.globalType.fieldNames.filter(f => !literalFields.contains(f))))
    }
  }

  def removeLiterals(mir: MatrixIR, literals: Map[String, IR]): MatrixIR = {
    if (literals.isEmpty)
      mir
    else {
      val literalFields = literals.keySet
      MatrixMapGlobals(mir,
        SelectFields(
          Ref("global", mir.typ.globalType),
          mir.typ.globalType.fieldNames.filter(f => !literalFields.contains(f))))
    }
  }

  def rewriteIR(ir: IR, newGlobalType: Type, literals: Map[String, IR]): IR = {
    val revMap = literals.map { case (id, literal) => (literal, id) }

    def rewrite(ir: IR, binds: Boolean): IR = {
      ir match {
        case Ref("global", t) if !binds => SelectFields(Ref("global", newGlobalType), t.asInstanceOf[TStruct].fieldNames)
        case _ => revMap.get(ir)
          .map(f => GetField(Ref("global", newGlobalType), f))
          .getOrElse(MapIR(rewrite(_, binds || Binds(ir, "global", 1)))(ir))
      }
    }
    rewrite(ir, binds = false)
  }

  def apply(ir: BaseIR): BaseIR = {
    MapIR.mapBaseIR(ir, {
      case MatrixMapRows(child, newRow) =>
        val literals = extractNonCompilable(newRow)
        val rewriteChild = addLiterals(child, literals)
        removeLiterals(
          MatrixMapRows(rewriteChild, rewriteIR(newRow, rewriteChild.typ.globalType, literals)),
          literals)
      case MatrixMapEntries(child, newEntries) =>
        val literals = extractNonCompilable(newEntries)
        val rewriteChild = addLiterals(child, literals)
        removeLiterals(
          MatrixMapEntries(rewriteChild, rewriteIR(newEntries, rewriteChild.typ.globalType, literals)),
          literals)
      case MatrixMapCols(child, newRow, newKey) =>
        val literals = extractNonCompilable(newRow)
        val rewriteChild = addLiterals(child, literals)
        removeLiterals(
          MatrixMapCols(rewriteChild, rewriteIR(newRow, rewriteChild.typ.globalType, literals), newKey),
          literals)
      case MatrixFilterRows(child, pred) =>
        val literals = extractNonCompilable(pred)
        val rewriteChild = addLiterals(child, literals)
        removeLiterals(
          MatrixFilterRows(rewriteChild, rewriteIR(pred, rewriteChild.typ.globalType, literals)),
          literals)
      case MatrixFilterCols(child, pred) =>
        val literals = extractNonCompilable(pred)
        val rewriteChild = addLiterals(child, literals)
        removeLiterals(
          MatrixFilterCols(rewriteChild, rewriteIR(pred, rewriteChild.typ.globalType, literals)),
          literals)
      case MatrixFilterEntries(child, pred) =>
        val literals = extractNonCompilable(pred)
        val rewriteChild = addLiterals(child, literals)
        removeLiterals(
          MatrixFilterEntries(rewriteChild, rewriteIR(pred, rewriteChild.typ.globalType, literals)),
          literals)
      case MatrixAggregateRowsByKey(child, entryAggIR, rowAggIR) =>
        val literals = extractNonCompilable(entryAggIR, rowAggIR)
        val rewriteChild = addLiterals(child, literals)
        removeLiterals(
          MatrixAggregateRowsByKey(rewriteChild,
            rewriteIR(entryAggIR, rewriteChild.typ.globalType, literals),
            rewriteIR(rowAggIR, rewriteChild.typ.globalType, literals)),
          literals)
      case MatrixAggregateColsByKey(child, entryAggIR, colAggIR) =>
        val literals = extractNonCompilable(entryAggIR, colAggIR)
        val rewriteChild = addLiterals(child, literals)
        removeLiterals(
          MatrixAggregateColsByKey(rewriteChild,
            rewriteIR(entryAggIR, rewriteChild.typ.globalType, literals),
            rewriteIR(colAggIR, rewriteChild.typ.globalType, literals)),
          literals)
      case TableFilter(child, pred) =>
        val literals = extractNonCompilable(pred)
        val rewriteChild = addLiterals(child, literals)
        removeLiterals(
          TableFilter(rewriteChild, rewriteIR(pred, rewriteChild.typ.globalType, literals)),
          literals)
      case TableMapRows(child, newRow) =>
        val literals = extractNonCompilable(newRow)
        val rewriteChild = addLiterals(child, literals)
        removeLiterals(
          TableMapRows(rewriteChild, rewriteIR(newRow, rewriteChild.typ.globalType, literals)),
          literals)
      case TableKeyByAndAggregate(child, expr, newKey, nPartitions, bufferSize) =>
        val literals = extractNonCompilable(expr, newKey)
        val rewriteChild = addLiterals(child, literals)
        removeLiterals(
          TableKeyByAndAggregate(rewriteChild, rewriteIR(expr, rewriteChild.typ.globalType, literals),
            rewriteIR(newKey, rewriteChild.typ.globalType, literals), nPartitions, bufferSize),
          literals)
      case TableAggregateByKey(child, expr) =>
        val literals = extractNonCompilable(expr)
        val rewriteChild = addLiterals(child, literals)
        removeLiterals(
          TableAggregateByKey(rewriteChild, rewriteIR(expr, rewriteChild.typ.globalType, literals)),
          literals)
      case TableAggregate(child, expr) =>
        val literals = extractNonCompilable(expr)
        val rewriteChild = addLiterals(child, literals)
        TableAggregate(rewriteChild, rewriteIR(expr, rewriteChild.typ.globalType, literals))
      case MatrixAggregate(child, expr) =>
        val literals = extractNonCompilable(expr)
        val rewriteChild = addLiterals(child, literals)
        MatrixAggregate(rewriteChild, rewriteIR(expr, rewriteChild.typ.globalType, literals))
      case ir => ir
    })
  }
}

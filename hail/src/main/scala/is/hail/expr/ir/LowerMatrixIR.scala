package is.hail.expr.ir

import is.hail.expr.types.{MatrixType, TInt32}
import is.hail.utils.FastSeq

object LowerMatrixIR {
  val entriesFieldName = "__entries"
  val colsFieldName = "__cols"

  def apply(ir: IR): IR = lower(ir)
  def apply(tir: TableIR): TableIR = lower(tir)
  def apply(mir: MatrixIR): TableIR = lower(mir)

  private[this] def lower(bir: BaseIR): BaseIR = bir match {
    case ir: IR => lower(ir)
    case tir: TableIR => lower(tir)
    case mir: MatrixIR => lower(mir)
  }

  private[this] def lower(ir: IR): IR =
    lowerChildren(ir).asInstanceOf[IR]

  private[this] def lower(tir: TableIR): TableIR =
    tableRules.applyOrElse(tir, (tir: TableIR) => lowerChildren(tir).asInstanceOf[TableIR])

  private[this] def lower(mir: MatrixIR): TableIR =
    matrixRules.applyOrElse(mir, (mir: MatrixIR) =>
      CastMatrixToTable(lowerChildren(mir).asInstanceOf[MatrixIR], entriesFieldName, colsFieldName))

  private[this] def lowerChildren(ir: BaseIR): BaseIR = {
    val loweredChildren = ir.children.map(lower)
    if ((ir.children, loweredChildren).zipped.forall(_ eq _))
      ir
    else {
      val newChildren = ir.children.zip(loweredChildren).map {
        case (_: MatrixIR, CastMatrixToTable(childMIR, _, _)) =>
          childMIR
        case (mir: MatrixIR, loweredChild: TableIR) =>
          CastTableToMatrix(
            loweredChild,
            entriesFieldName,
            colsFieldName,
            mir.typ.colKey)
        case (_, loweredChild) =>
          loweredChild
      }
      ir.copy(newChildren)
    }
  }

  def colVals(tir: TableIR): IR =
    GetField(Ref("global", tir.typ.globalType), colsFieldName)

  def globals(tir: TableIR): IR =
    SelectFields(
      Ref("global", tir.typ.globalType),
      tir.typ.globalType.fieldNames.diff(FastSeq(colsFieldName)))

  def nCols(tir: TableIR): IR = ArrayLen(colVals(tir))

  def entries(tir: TableIR): IR =
    GetField(Ref("row", tir.typ.rowType), entriesFieldName)

  private[this] def matrixRules: PartialFunction[MatrixIR, TableIR] = {
    case MatrixKeyRowsBy(child, keys, isSorted) =>
      TableKeyBy(lower(child), keys, isSorted)

    case MatrixFilterRows(child, pred) =>
      val lowered = lower(child)
      TableRename(
        TableFilter(
          TableRename(lower(child), Map(entriesFieldName -> MatrixType.entriesIdentifier), Map.empty),
          Let("va", Ref("row", lowered.typ.rowType), pred)),
        Map(MatrixType.entriesIdentifier -> entriesFieldName), Map.empty)

    case MatrixMapGlobals(child, newGlobals) =>
      val lowered = lower(child)
      val loweredNewGlobals =
        Let("global", globals(lowered), newGlobals)

      TableMapGlobals(
        lowered,
        InsertFields(loweredNewGlobals, FastSeq(colsFieldName -> colVals(lowered))))

    case MatrixFilterEntries(child, pred) =>
      val lowered = lower(child)
      val newEntries =
        ArrayMap(
          ArrayRange(I32(0), nCols(lowered), I32(1)),
          "i",
          Let("g", ArrayRef(entries(lowered), Ref("i", TInt32())),
            If(
              Let("sa", ArrayRef(colVals(lowered), Ref("i", TInt32())),
                pred),
              Ref("g", child.typ.entryType),
              NA(child.typ.entryType))))
      val newRow = InsertFields(
        Ref("row", lowered.typ.rowType),
        FastSeq(entriesFieldName -> newEntries))

      TableMapRows(lowered, newRow)
  }

  private[this] def tableRules: PartialFunction[TableIR, TableIR] = {
    case CastMatrixToTable(child, entries, cols) =>
      CastMatrixToTable(
        CastTableToMatrix(lower(child), entriesFieldName, colsFieldName, child.typ.colKey),
        entries,
        cols)

    case MatrixRowsTable(child) =>
      val lowered = lower(child)
      val dropCols =
        TableMapGlobals(
          lowered,
          SelectFields(
            Ref("global", lowered.typ.globalType),
            child.typ.globalType.fieldNames))
      TableMapRows(
        dropCols,
        SelectFields(
          Ref("row", lowered.typ.rowType),
          child.typ.rowType.fieldNames))
  }
}

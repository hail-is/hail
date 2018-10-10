package is.hail.expr.ir

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

  private[this] def matrixRules: PartialFunction[MatrixIR, TableIR] = {
    case MatrixKeyRowsBy(child, keys, isSorted) =>
      TableKeyBy(lower(child), keys, isSorted)
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

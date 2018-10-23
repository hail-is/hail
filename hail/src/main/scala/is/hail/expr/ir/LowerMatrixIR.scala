package is.hail.expr.ir

import is.hail.expr.types._
import is.hail.utils.FastSeq

object LowerMatrixIR {
  val entriesFieldName = "__entries"
  val colsFieldName = "__cols"
  val colsField = Symbol(colsFieldName)
  val entriesField = Symbol(entriesFieldName)

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

  import is.hail.expr.ir.IRBuilder._

  private[this] def matrixRules: PartialFunction[MatrixIR, TableIR] = {
    case MatrixKeyRowsBy(child, keys, isSorted) =>
      lower(child).keyBy(keys, isSorted)

    case MatrixFilterRows(child, pred) =>
      lower(child)
        .rename(Map(entriesFieldName -> MatrixType.entriesIdentifier))
        .filter(let ('va <=: 'row) in pred)
        .rename(Map(MatrixType.entriesIdentifier -> entriesFieldName))

    case MatrixFilterCols(child, pred) =>
      lower(child)
        .mapGlobals('global.insertFields('newColIdx ->
          irRange(0, 'global(colsField).len)
            .filter('i ~>
              (let ('sa <=: 'global(colsField)('i),
                    'global <=: 'global.dropFields(colsField))
                in pred))))
        .mapRows('row.insertFields(entriesField -> 'global('newColIdx).map('i ~> 'row(entriesField)('i))))
        .mapGlobals('global
          .insertFields(colsField ->
            'global('newColIdx).map('i ~> 'global(colsField)('i)))
          .dropFields('newColIdx))

    case MatrixChooseCols(child, oldIndices) =>
      lower(child)
        .mapGlobals('global.insertFields('newColIdx -> oldIndices.map(I32)))
        .mapRows('row.insertFields(entriesField -> 'global('newColIdx).map('i ~> 'row(entriesField)('i))))
        .mapGlobals('global.insertFields(colsField -> 'global('newColIdx).map('i ~> 'global(colsField)('i))))

    case MatrixMapGlobals(child, newGlobals) =>
      lower(child)
        .mapGlobals(
          (let ('global <=: 'global.dropFields(colsField)) in newGlobals)
          .insertFields(colsField -> 'global(colsField)))

    case MatrixFilterEntries(child, pred) =>
      lower(child).mapRows('row.insertFields(entriesField ->
        irRange(0, 'global(colsField).len).map { 'i ~>
          (let ('g <=: 'row(entriesField)('i))
            in irIf (let ('sa <=: 'global(colsField)('i),
                          'va <=: 'row.dropFields(entriesField),
                          'global <=: 'global.dropFields(colsField))
                      in pred)
              { 'g }{ NA(child.typ.entryType) })}))
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

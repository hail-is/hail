package is.hail.expr.ir

import is.hail.expr.types._
import is.hail.expr.types.virtual.{TArray, TInt32}
import is.hail.utils._

object LowerMatrixIR {
  val entriesFieldName = MatrixType.entriesIdentifier
  val colsFieldName = "__cols"
  val colsField = Symbol(colsFieldName)
  val entriesField = Symbol(entriesFieldName)

  def apply(ir: IR): IR = lower(ir)
  def apply(tir: TableIR): TableIR = lower(tir)
  def apply(mir: MatrixIR): MatrixIR =
    CastTableToMatrix(lower(mir),
      entriesFieldName,
      colsFieldName,
      mir.typ.colKey)

  private[this] def lower(bir: BaseIR): BaseIR = bir match {
    case ir: IR => lower(ir)
    case tir: TableIR => lower(tir)
    case mir: MatrixIR => lower(mir)
  }

  private[this] def lower(ir: IR): IR = {
    val lowered = lowerChildren(ir).asInstanceOf[IR]
    assert(lowered.typ == ir.typ)
    lowered
  }

  private[this] def lower(tir: TableIR): TableIR = {
    val lowered = tableRules.applyOrElse(tir, (tir: TableIR) => lowerChildren(tir).asInstanceOf[TableIR])
    if(lowered.typ != tir.typ)
      fatal(s"lowering changed type:\n  before: ${tir.typ}\n  after: ${lowered.typ}")
    lowered
  }

  private[this] def lower(mir: MatrixIR): TableIR = {
    val lowered = matrixRules.applyOrElse(mir, (mir: MatrixIR) =>
      CastMatrixToTable(lowerChildren(mir).asInstanceOf[MatrixIR], entriesFieldName, colsFieldName))
    assert(lowered.typ == loweredType(mir.typ), s"\n  ${lowered.typ}\n  ${loweredType(mir.typ)}")
    lowered
  }

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

  def loweredType(
    typ: MatrixType,
    entriesFieldName: String = entriesFieldName,
    colsFieldName: String = colsFieldName
  ): TableType = TableType(
    rowType = typ.rvRowType.rename(Map(MatrixType.entriesIdentifier -> entriesFieldName)),
    key = typ.rowKey,
    globalType = typ.globalType.appendKey(colsFieldName, TArray(typ.colType)))

  import is.hail.expr.ir.IRBuilder._

  private[this] def matrixRules: PartialFunction[MatrixIR, TableIR] = {
    case CastTableToMatrix(child, entries, cols, colKey) =>
      TableRename(lower(child), Map(entries -> entriesFieldName), Map(cols -> colsFieldName))

    case MatrixKeyRowsBy(child, keys, isSorted) =>
      lower(child).keyBy(keys, isSorted)

    case MatrixFilterRows(child, pred) =>
      lower(child)
        .rename(Map(entriesFieldName -> MatrixType.entriesIdentifier))
        .filter(let (va = 'row,
                     global = 'global.dropFields(colsField))
                in pred)
        .rename(Map(MatrixType.entriesIdentifier -> entriesFieldName))

    case MatrixFilterCols(child, pred) =>
      lower(child)
        .mapGlobals('global.insertFields('newColIdx ->
          irRange(0, 'global(colsField).len)
            .filter('i ~>
              (let (sa = 'global(colsField)('i),
                    global = 'global.dropFields(colsField))
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
        .mapGlobals('global
          .insertFields(colsField -> 'global('newColIdx).map('i ~> 'global(colsField)('i)))
          .dropFields('newColIdx))

    case MatrixMapGlobals(child, newGlobals) =>
      lower(child)
        .mapGlobals(
          let (global = 'global.dropFields(colsField)) { newGlobals }
          .insertFields(colsField -> 'global(colsField)))

    case MatrixFilterEntries(child, pred) =>
      lower(child).mapRows('row.insertFields(entriesField ->
        irRange(0, 'global(colsField).len).map { 'i ~>
          let (g = 'row(entriesField)('i)) {
            irIf (let (sa = 'global (colsField)('i),
                       va = 'row,
                       global = 'global.dropFields(colsField))
                   in !irToProxy(pred)) {
              NA(child.typ.entryType)
            } {
              'g
            }
          }
        }))

    case MatrixMapEntries(child, newEntries) =>
      lower(child).mapRows('row.insertFields(entriesField ->
        irRange(0, 'global(colsField).len).map { 'i ~>
          let (g = 'row(entriesField)('i),
               sa = 'global(colsField)('i),
               va = 'row,
               global = 'global.dropFields(colsField)) {
            newEntries
          }
        }))

    case MatrixRepartition(child, n, shuffle) => TableRepartition(lower(child), n, shuffle)

    case MatrixUnionRows(children) =>
      // FIXME: this should check that all children have the same column keys.
      TableUnion(MatrixUnionRows.unify(children).map(lower))

    case MatrixDistinctByRow(child) => TableDistinct(lower(child))

    case MatrixCollectColsByKey(child) =>
      lower(child)
        .mapGlobals('global.insertFields('newColIdx ->
          irRange(0, 'global(colsField).len).map { 'i ~>
            makeTuple('global(colsField)('i).selectFields(child.typ.colKey: _*),
                      'i)
          }.groupByKey.toArray))
        .mapRows('row.insertFields(entriesField ->
          'global('newColIdx).map { 'kv ~>
            makeStruct(child.typ.entryType.fieldNames.map { s =>
              (Symbol(s), 'kv('value).map { 'i ~> 'row(entriesField)('i)(Symbol(s))})}: _*)
          }))
        .mapGlobals('global
          .insertFields(colsField ->
            'global('newColIdx).map { 'kv ~>
              'kv('key).insertFields(
                child.typ.colValueStruct.fieldNames.map { s =>
                  (Symbol(s), 'kv('value).map('i ~> 'global(colsField)('i)(Symbol(s))))}: _*)
            })
          .dropFields('newColIdx)
        )

    case mr: MatrixRead if mr.reader.canLower => mr.reader.lower(mr)
  }

  private[this] def tableRules: PartialFunction[TableIR, TableIR] = {
    case CastMatrixToTable(child, entries, cols) =>
      TableRename(lower(child), Map(entriesFieldName -> entries), Map(colsFieldName -> cols))

    case MatrixRowsTable(child) =>
      lower(child)
        .mapGlobals('global.dropFields(colsField))
        .mapRows('row.dropFields(entriesField))

    case MatrixColsTable(child) =>
      val colKey = child.typ.colKey
      let(__cols_and_globals = lower(child).getGlobals) {
        val sortedCols = if (colKey.isEmpty)
          '__cols_and_globals (colsField)
        else
          irRange(0, irArrayLen('__cols_and_globals (colsField)), 1)
            .map {
              'i ~> let(__cols_element = '__cols_and_globals (colsField)('i)) {
                makeStruct(
                  // key struct
                  '_1 -> '__cols_element.selectFields(colKey: _*),
                  '_2 -> '__cols_element)
              }
            }
            .sort(true, onKey = true)
            .map {
              'elt ~> 'elt ('_2)
            }
        makeStruct('rows -> sortedCols, 'global -> '__cols_and_globals.dropFields(colsField))
      }.parallelize(None).keyBy(child.typ.colKey)
  }
}

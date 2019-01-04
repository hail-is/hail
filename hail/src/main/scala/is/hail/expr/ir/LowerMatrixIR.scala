package is.hail.expr.ir

import is.hail.expr.types._
import is.hail.expr.types.virtual.{TArray, TInt32}
import is.hail.utils._

object LowerMatrixIR {

  def apply(ir: IR): IR = lower(ir)
  def apply(tir: TableIR): TableIR = lower(tir)
  def apply(mir: MatrixIR): MatrixIR =
    CastTableToMatrix(lower(mir),
      EntriesSym,
      ColsSym,
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
      CastMatrixToTable(lowerChildren(mir).asInstanceOf[MatrixIR], EntriesSym, ColsSym))
    assert(lowered.typ == loweredType(mir.typ), lowered.typ + "\n" + loweredType(mir.typ))
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
            EntriesSym,
            ColsSym,
            mir.typ.colKey)
        case (_, loweredChild) =>
          loweredChild
      }
      ir.copy(newChildren)
    }
  }

  def colVals(tir: TableIR): IR =
    GetField(Ref(GlobalSym, tir.typ.globalType), ColsSym)

  def globals(tir: TableIR): IR =
    SelectFields(
      Ref(GlobalSym, tir.typ.globalType),
      tir.typ.globalType.fieldNames.diff(FastSeq(ColsSym)))

  def nCols(tir: TableIR): IR = ArrayLen(colVals(tir))

  def entries(tir: TableIR): IR =
    GetField(Ref(RowSym, tir.typ.rowType), EntriesSym)

  def loweredType(
    typ: MatrixType,
    entries: Sym = EntriesSym,
    cols: Sym = ColsSym
  ): TableType = TableType(
    rowType = typ.rvRowType.rename(Map(EntriesSym -> entries)),
    key = typ.rowKey,
    globalType = typ.globalType.appendKey(cols, TArray(typ.colType)))

  import is.hail.expr.ir.IRBuilder._

  private[this] def matrixRules: PartialFunction[MatrixIR, TableIR] = {
    case CastTableToMatrix(child, entries, cols, colKey) =>
      TableRename(lower(child), Map(entries -> EntriesSym), Map(cols -> ColsSym))

    case MatrixKeyRowsBy(child, keys, isSorted) =>
      lower(child).keyBy(keys, isSorted)

    case MatrixFilterRows(child, pred) =>
      lower(child)
        .rename(Map(EntriesSym -> EntriesSym))
        .filter(let (RowSym ~> RowSym,
                     GlobalSym ~> GlobalSym.dropFields(ColsSym))
                in pred)
        .rename(Map(EntriesSym -> EntriesSym))

    case MatrixFilterCols(child, pred) =>
      val newColIdx = genSym("newColIdx")
      val i = genSym("i")
      lower(child)
        .mapGlobals(GlobalSym.insertFields(newColIdx ->
          irRange(0, GlobalSym(ColsSym).len)
            .filter(i ~>
              (let (ColSym ~> GlobalSym(ColsSym)(i),
                    GlobalSym ~> GlobalSym.dropFields(ColsSym))
                in pred))))
        .mapRows(RowSym.insertFields(EntriesSym -> GlobalSym(newColIdx).map(i ~> RowSym(EntriesSym)(i))))
        .mapGlobals(GlobalSym
          .insertFields(ColsSym ->
            GlobalSym(newColIdx).map(i ~> GlobalSym(ColsSym)(i)))
          .dropFields(newColIdx))

    case MatrixChooseCols(child, oldIndices) =>
      val newColIdx = genSym("newColIdx")
      val i = genSym("i")
      lower(child)
        .mapGlobals(GlobalSym.insertFields(newColIdx -> oldIndices.map(I32)))
        .mapRows(RowSym.insertFields(EntriesSym -> GlobalSym(newColIdx).map(i ~> RowSym(EntriesSym)(i))))
        .mapGlobals(GlobalSym
          .insertFields(ColsSym -> GlobalSym(newColIdx).map(i ~> GlobalSym(ColsSym)(i)))
          .dropFields(newColIdx))

    case MatrixMapGlobals(child, newGlobals) =>
      lower(child)
        .mapGlobals(
          let (GlobalSym ~> GlobalSym.dropFields(ColsSym)) { newGlobals }
          .insertFields(ColsSym -> GlobalSym(ColsSym)))

    case MatrixFilterEntries(child, pred) =>
      val i = genSym("i")
      lower(child).mapRows(RowSym.insertFields(EntriesSym ->
        irRange(0, GlobalSym(ColsSym).len).map { i ~>
          let (EntrySym ~> RowSym(EntriesSym)(i)) {
            irIf (let (ColSym ~> GlobalSym (ColsSym)(i),
                       RowSym ~> RowSym,
                       GlobalSym ~> GlobalSym.dropFields(ColsSym))
                   in !irToProxy(pred)) {
              NA(child.typ.entryType)
            } {
              EntrySym
            }
          }
        }))

    case MatrixUnionCols(left, right) =>
      val a = genSym("a")
      val rightEntries = genSym("rightEntries")
      val rightCols = genSym("rightCols")
      TableJoin(
        lower(left),
        lower(right)
          .mapRows(RowSym
            .insertFields(rightEntries -> RowSym(EntriesSym))
            .selectFields(right.typ.rowKey :+ rightEntries: _*))
          .mapGlobals(GlobalSym
            .insertFields(rightCols -> GlobalSym(ColsSym))
            .selectFields(rightCols)),
        "inner")
        .mapRows(RowSym
          .insertFields(EntriesSym ->
            makeArray(RowSym(EntriesSym), RowSym(rightEntries)).flatMap(a ~> a))
          // TableJoin puts keys first; drop rightEntries, but also restore left row field order
          .selectFields(left.typ.rvRowType.fieldNames: _*))
        .mapGlobals(GlobalSym
          .insertFields(ColsSym ->
            makeArray(GlobalSym(ColsSym), GlobalSym(rightCols)).flatMap(a ~> a))
          .dropFields(rightCols))

    case MatrixMapEntries(child, newEntries) =>
      val i = genSym("i")
      lower(child).mapRows(RowSym.insertFields(EntriesSym ->
        irRange(0, GlobalSym(ColsSym).len).map { i ~>
          let (EntrySym ~> RowSym(EntriesSym)(i),
               ColSym ~> GlobalSym(ColsSym)(i),
               RowSym ~> RowSym,
               GlobalSym ~> GlobalSym.dropFields(ColsSym)) {
            newEntries
          }
        }))

    case MatrixRepartition(child, n, shuffle) => TableRepartition(lower(child), n, shuffle)

    case MatrixUnionRows(children) =>
      // FIXME: this should check that all children have the same column keys.
      TableUnion(MatrixUnionRows.unify(children).map(lower))

    case MatrixDistinctByRow(child) => TableDistinct(lower(child))

    case MatrixCollectColsByKey(child) =>
      val newColIdx = genSym("newColIdx")
      val i = genSym("i")
      val kv = genSym("kv")
      lower(child)
        .mapGlobals(GlobalSym.insertFields(newColIdx ->
          irRange(0, GlobalSym(ColsSym).len).map { i ~>
            makeTuple(GlobalSym(ColsSym)(i).selectFields(child.typ.colKey: _*),
                      i)
          }.groupByKey.toArray))
        .mapRows(RowSym.insertFields(EntriesSym ->
          GlobalSym(newColIdx).map { kv ~>
            makeStruct(child.typ.entryType.fieldNames.map { s =>
              (s, kv(I("value")).map { i ~> RowSym(EntriesSym)(i)(s) }) }: _*)
          }))
        .mapGlobals(GlobalSym
          .insertFields(ColsSym ->
            GlobalSym(newColIdx).map { kv ~>
              kv(I("key")).insertFields(
                child.typ.colValueStruct.fieldNames.map { s =>
                  (s, kv(I("value")).map(i ~> GlobalSym(ColsSym)(i)(s)))}: _*)
            })
          .dropFields(newColIdx)
        )

    case MatrixExplodeRows(child, path) => TableExplode(lower(child), path)
  }

  private[this] def tableRules: PartialFunction[TableIR, TableIR] = {
    case CastMatrixToTable(child, entries, cols) =>
      TableRename(lower(child), Map(EntriesSym -> entries), Map(ColsSym -> cols))

    case MatrixRowsTable(child) =>
      lower(child)
        .mapGlobals(GlobalSym.dropFields(ColsSym))
        .mapRows(RowSym.dropFields(EntriesSym))

    case MatrixColsTable(child) =>
      val i = genSym("i")
      val colElem = genSym("colElem")
      val elt = genSym("elt")
      val colKey = child.typ.colKey
      let(GlobalAndColsSym ~> lower(child).getGlobals) {
        val sortedCols = if (colKey.isEmpty)
          GlobalAndColsSym(ColsSym)
        else
          irRange(0, irArrayLen(GlobalAndColsSym(ColsSym)), 1)
            .map {
              i ~> let(colElem ~> GlobalAndColsSym(ColsSym)(i)) {
                makeStruct(
                  // key struct
                  Identifier("_1") -> colElem.selectFields(colKey: _*),
                  Identifier("_2") -> colElem)
              }
            }
            .sort(true, onKey = true)
            .map {
              elt ~> elt (Identifier("_2"))
            }
        makeStruct(RowsSym -> sortedCols, GlobalSym -> GlobalAndColsSym.dropFields(ColsSym))
      }.parallelize(None).keyBy(child.typ.colKey)
  }
}

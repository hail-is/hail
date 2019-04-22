package is.hail.expr.ir

import is.hail.expr.ir.functions.{WrappedMatrixToMatrixFunction, WrappedMatrixToTableFunction, WrappedMatrixToValueFunction}
import is.hail.expr.types._
import is.hail.expr.types.virtual.{TArray, TInt32, TInterval}
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
    case bmir: BlockMatrixIR => bmir
  }

  private[this] def lower(ir: IR): IR = {
    val lowered = valueRules.applyOrElse(ir, (ir: IR) => lowerChildren(ir).asInstanceOf[IR])
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
    assert(lowered.typ == loweredType(mir.typ), s"\n  ACTUAL: ${ lowered.typ }\n  EXPECT: ${ loweredType(mir.typ) }" +
      s"\n  BEFORE: ${ Pretty(mir) }\n  AFTER: ${ Pretty(lowered) }")
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

    case MatrixToMatrixApply(child, function) =>
      val loweredChild = lower(child)
      TableToTableApply(loweredChild, function.lower()
        .getOrElse(WrappedMatrixToMatrixFunction(function,
          colsFieldName, colsFieldName,
          entriesFieldName, entriesFieldName,
          child.typ.colKey)))

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

    case MatrixAnnotateRowsTable(child, table, root, product) =>
      val kt = table.typ.keyType
      if (kt.size == 1 && kt.types(0) == TInterval(child.typ.rowKeyStruct.types(0)))
        TableIntervalJoin(lower(child), lower(table), root, product)
      else
        TableLeftJoinRightDistinct(lower(child), lower(table), root)

    case MatrixChooseCols(child, oldIndices) =>
      lower(child)
        .mapGlobals('global.insertFields('newColIdx -> oldIndices.map(I32)))
        .mapRows('row.insertFields(entriesField -> 'global('newColIdx).map('i ~> 'row(entriesField)('i))))
        .mapGlobals('global
          .insertFields(colsField -> 'global('newColIdx).map('i ~> 'global(colsField)('i)))
          .dropFields('newColIdx))

    case MatrixAnnotateColsTable(child, table, root) =>
      val col = Symbol(genUID())
      val colKey = makeStruct(table.typ.key.zip(child.typ.colKey).map { case (tk, mck) => Symbol(tk) -> col(Symbol(mck))}: _*)
      lower(child)
        .mapGlobals(let(__dictfield = lower(table)
          .keyBy(FastIndexedSeq())
          .collect()
          .apply('rows)
          .arrayStructToDict(table.typ.key)) {
          'global.insertFields(colsField ->
            'global(colsField).map(col ~> col.insertFields(Symbol(root) -> '__dictfield.invoke("get", colKey))))
        })

    case MatrixMapGlobals(child, newGlobals) =>
      lower(child)
        .mapGlobals(
          let (global = 'global.dropFields(colsField)) { newGlobals }
          .insertFields(colsField -> 'global(colsField)))

    case MatrixMapRows(child, newRow) => {
      def liftScans(ir: IR): IRProxy = {
        val scans = new ArrayBuilder[(String, IR)]

        def f(ir: IR): IR = ir match {
          case x@(_: ApplyScanOp | _: AggFilter | _: AggExplode | _: AggGroupBy) if ContainsScan(x) =>
            assert(!ContainsAgg(x))
            val s = genUID()
            scans += (s -> x)
            Ref(s, x.typ)
          case _ =>
            MapIR(f)(ir)
        }

        val b0 = f(ir)

        val b: IRProxy =
          if (ContainsAgg(b0)) {
            irRange(0, 'row (entriesField).len)
              .filter('i ~> !'row (entriesField)('i).isNA)
              .arrayAgg('i ~>
                (aggLet(sa = 'global (colsField)('i),
                  g = 'row (entriesField)('i))
                  in b0))
          } else
            b0

        scans.result().foldLeft(b) { case (acc, (s, x)) =>
          (env: E) => {
            Let(s, x, acc(env))
          }
        }
      }

      val lc = lower(child)
      val e = Env[IR]("va" -> Ref("row", lc.typ.rowType),
        "global" -> SelectFields(Ref("global", lc.typ.globalType), child.typ.globalType.fieldNames))
      lc.mapRows(
        liftScans(Subst(newRow, BindingEnv(e, scan = Some(e), agg = Some(e))))
          .insertFields(entriesField -> 'row (entriesField)))
    }

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

    case MatrixUnionCols(left, right) =>
      val rightEntries = genUID()
      val rightCols = genUID()
      TableJoin(
        lower(left),
        lower(right)
          .mapRows('row
            .insertFields(Symbol(rightEntries) -> 'row(entriesField))
            .selectFields(right.typ.rowKey :+ rightEntries: _*))
          .mapGlobals('global
            .insertFields(Symbol(rightCols) -> 'global(colsField))
            .selectFields(rightCols)),
        "inner")
        .mapRows('row
          .insertFields(entriesField ->
            makeArray('row(entriesField), 'row(Symbol(rightEntries))).flatMap('a ~> 'a))
          // TableJoin puts keys first; drop rightEntries, but also restore left row field order
          .selectFields(left.typ.rvRowType.fieldNames: _*))
        .mapGlobals('global
          .insertFields(colsField ->
            makeArray('global(colsField), 'global(Symbol(rightCols))).flatMap('a ~> 'a))
          .dropFields(Symbol(rightCols)))

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

    case MatrixRowsHead(child, n) => TableHead(lower(child), n)

    case MatrixExplodeCols(child, path) =>
      val loweredChild = lower(child)
      val lengths = Symbol(genUID())
      val colIdx = Symbol(genUID())
      val nestedIdx = Symbol(genUID())
      val colElementUID1 = Symbol(genUID())


      val nestedRefs = path.init.scanLeft('global(colsField)(colIdx): IRProxy)((irp, name) => irp(Symbol(name)))
      val postExplodeSelector = path.zip(nestedRefs).zipWithIndex.foldRight[IRProxy](nestedIdx) {
        case (((field, ref), i), arg) =>
          ref.insertFields(Symbol(field) ->
            (if (i == nestedRefs.length - 1)
              ref(Symbol(field)).toArray(arg)
            else
              arg))
      }

      val arrayIR = path.foldLeft[IRProxy](colElementUID1) { case (irp, fieldName) => irp(Symbol(fieldName)) }
      loweredChild
        .mapGlobals('global.insertFields(lengths -> 'global(colsField).map( { colElementUID1 ~> arrayIR.len.orElse(0)})))
        .mapGlobals('global.insertFields(colsField ->
          irRange(0, 'global(colsField).len, 1)
            .flatMap( { colIdx ~>
                irRange(0, 'global(lengths)(colIdx), 1)
                .map( { nestedIdx ~> postExplodeSelector })
              })))
        .mapRows('row.insertFields(entriesField ->
          irRange(0, 'row(entriesField).len, 1)
            .flatMap(colIdx ~>
              irRange(0, 'global(lengths)(colIdx), 1).map(Symbol(genUID()) ~> 'row(entriesField)(colIdx)))))
        .mapGlobals('global.dropFields(lengths))

    case MatrixAggregateRowsByKey(child, entryExpr, rowExpr) =>
      lower(child)
        .aggregateByKey(
          aggLet(va = 'row) {
            rowExpr.insertFields(entriesField -> irRange(0, 'global (colsField).len)
              .aggElements('__element_idx, '__result_idx)(
                  let(sa = 'global(colsField)('__result_idx)) {
                    aggLet(sa = 'global (colsField)('__element_idx),
                      g = 'row (entriesField)('__element_idx)) {
                      entryExpr
                    }
                  }))})

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
              (Symbol(s), 'kv('value).map { 'i ~> 'row(entriesField)('i)(Symbol(s)) }) }: _*)
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

    case MatrixExplodeRows(child, path) => TableExplode(lower(child), path)

    case mr: MatrixRead => mr.lower()

    case MatrixAggregateColsByKey(child, entryExpr, colExpr) =>
      val colKey = child.typ.colKey

      val originalColIdx = Symbol(genUID())
      val newColIdx1 = Symbol(genUID())
      val newColIdx2 = Symbol(genUID())
      val colsAggIdx = Symbol(genUID())
      val keyMap = Symbol(genUID())
      val aggElementIdx = Symbol(genUID())
      lower(child)
        .mapGlobals('global.insertFields(keyMap ->
          let(__cols_field = 'global (colsField)) {
            irRange(0, '__cols_field.len)
              .map(originalColIdx ~> let(__cols_field_element = '__cols_field (originalColIdx)) {
                makeStruct('key -> '__cols_field_element.selectFields(colKey: _*), 'value -> originalColIdx)
              })
              .groupByKey
              .toArray
          }))
        .mapRows('row.insertFields(entriesField ->
          let(__entries = 'row (entriesField), __key_map = 'global (keyMap), va = 'row) {
            irRange(0, '__key_map.len)
              .map(newColIdx1 ~> '__key_map (newColIdx1)
                .apply('value)
                .arrayAgg(aggElementIdx ~>
                  aggLet(va = 'row, g = '__entries (aggElementIdx), sa = 'global (colsField)(aggElementIdx)) {
                    entryExpr
                  }))}))
        .mapGlobals(
          'global.insertFields(colsField ->
            let(__key_map = 'global (keyMap)) {
              irRange(0, '__key_map.len)
                .map(newColIdx2 ~>
                  concatStructs(
                    '__key_map (newColIdx2)('key),
                    '__key_map (newColIdx2)('value)
                      .arrayAgg(colsAggIdx ~> aggLet(sa = 'global (colsField)(colsAggIdx)) {
                        colExpr
                      })
                  )
                )
            }
          ).dropFields(keyMap))
  }

  private[this] def tableRules: PartialFunction[TableIR, TableIR] = {
    case CastMatrixToTable(child, entries, cols) =>
      TableRename(lower(child), Map(entriesFieldName -> entries), Map(colsFieldName -> cols))

    case MatrixEntriesTable(child) =>
      val oldColIdx = Symbol(genUID())
      val lambdaIdx1 = Symbol(genUID())
      val currentColIdx = Symbol(genUID())
      lower(child)
        .mapGlobals('global.insertFields(oldColIdx ->
          irRange(0, 'global (colsField).len)
            .map(lambdaIdx1 ~> makeStruct('key -> 'global (colsField)(lambdaIdx1).selectFields(child.typ.colKey: _*), 'value -> lambdaIdx1))
            .sort(ascending = true, onKey = true)
            .map(lambdaIdx1 ~> lambdaIdx1('value))))
        .mapRows('row.insertFields(currentColIdx -> 'global (oldColIdx)
          .filter(lambdaIdx1 ~> !'row (entriesField)(lambdaIdx1).isNA)))
        .explode(currentColIdx)
        .mapRows(let(
          __current_idx = 'row (currentColIdx),
          __col_struct = 'global (colsField)('__current_idx),
          __entry_struct = 'row (entriesField)('__current_idx)) {
          val newFields = child.typ.colType.fieldNames.map(Symbol(_)).map(f => f -> '__col_struct (f)) ++
            child.typ.entryType.fieldNames.map(Symbol(_)).map(f => f -> '__entry_struct (f))
          'row
            .dropFields(entriesField, currentColIdx)
            .insertFields(newFields: _*)
        }).mapGlobals('global.dropFields(colsField, oldColIdx))
        .keyBy(child.typ.rowKey ++ child.typ.colKey, isSorted = !(child.typ.rowKey.isEmpty && child.typ.colKey.nonEmpty))

    case MatrixToTableApply(child, function) =>
      val loweredChild = lower(child)
      TableToTableApply(loweredChild,
        function.lower()
          .getOrElse(WrappedMatrixToTableFunction(function, colsFieldName, entriesFieldName, child.typ.colKey)))

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

  private[this] def valueRules: PartialFunction[IR, IR] = {
    case MatrixToValueApply(child, function) => TableToValueApply(lower(child), function.lower()
      .getOrElse(WrappedMatrixToValueFunction(function, colsFieldName, entriesFieldName, child.typ.colKey)))
    case MatrixWrite(child, writer) =>
      TableWrite(lower(child), WrappedMatrixWriter(writer, colsFieldName, entriesFieldName, child.typ.colKey))
    case MatrixAggregate(child, query) =>
      val idx = Symbol(genUID())
      lower(child)
        .aggregate(
          aggLet(
            __entries_field = 'row (entriesField),
            __cols_field = 'global (colsField)) {
            irRange(0, '__entries_field.len)
              .filter(idx ~> !'__entries_field (idx).isNA)
              .aggExplode(idx ~> aggLet(va = 'row, sa = '__cols_field (idx), g = '__entries_field (idx)) {
                query
              })
          })
  }
}

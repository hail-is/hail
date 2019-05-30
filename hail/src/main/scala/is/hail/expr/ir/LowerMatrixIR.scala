package is.hail.expr.ir

import is.hail.expr.ir.functions.{WrappedMatrixToMatrixFunction, WrappedMatrixToTableFunction, WrappedMatrixToValueFunction}
import is.hail.expr.types._
import is.hail.expr.types.virtual.{TArray, TInt32, TInterval}
import is.hail.utils._

object LowerMatrixIR {
  val entriesFieldName: String = MatrixType.entriesIdentifier
  val colsFieldName: String = "__cols"
  val colsField: Symbol = Symbol(colsFieldName)
  val entriesField: Symbol = Symbol(entriesFieldName)

  def apply(ir: IR): IR = {
    val ab = new ArrayBuilder[(String, IR)]
    val l1 = lower(ir, ab)
    ab.result().foldRight[IR](l1) { case ((ident, value), body) => RelationalLet(ident, value, body) }
  }

  def apply(tir: TableIR): TableIR = {
    val ab = new ArrayBuilder[(String, IR)]
    val l1 = lower(tir, ab)
    ab.result().foldRight[TableIR](l1) { case ((ident, value), body) => RelationalLetTable(ident, value, body) }
  }

  def apply(mir: MatrixIR): TableIR = {
    val ab = new ArrayBuilder[(String, IR)]

    val l1 = lower(mir, ab)
    ab.result().foldRight[TableIR](l1) { case ((ident, value), body) => RelationalLetTable(ident, value, body) }
  }


  private[this] def lowerChildren(ir: BaseIR, ab: ArrayBuilder[(String, IR)]): BaseIR = {
    val loweredChildren = ir.children.map {
      case tir: TableIR => lower(tir, ab)
      case mir: MatrixIR => throw new RuntimeException(s"expect specialized lowering rule for " +
        s"${ ir.getClass.getName }\n  Found MatrixIR child $mir")
      case bmir: BlockMatrixIR => bmir // FIXME wrong
      case vir: IR => lower(vir, ab)
    }
    if ((ir.children, loweredChildren).zipped.forall(_ eq _))
      ir
    else
      ir.copy(loweredChildren)
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

  private[this] def lower(mir: MatrixIR, ab: ArrayBuilder[(String, IR)]): TableIR = {
    val lowered = mir match {
      case RelationalLetMatrixTable(name, value, body) =>
        RelationalLetTable(name, lower(value, ab), lower(body, ab))

      case CastTableToMatrix(child, entries, cols, colKey) =>
        TableRename(lower(child, ab), Map(entries -> entriesFieldName), Map(cols -> colsFieldName))

      case MatrixToMatrixApply(child, function) =>
        val loweredChild = lower(child, ab)
        TableToTableApply(loweredChild, function.lower()
          .getOrElse(WrappedMatrixToMatrixFunction(function,
            colsFieldName, colsFieldName,
            entriesFieldName, entriesFieldName,
            child.typ.colKey)))

      case MatrixRename(child, globalMap, colMap, rowMap, entryMap) =>
        var t = lower(child, ab).rename(rowMap, globalMap)

        if (colMap.nonEmpty) {
          val newColsType = TArray(child.typ.colType.rename(colMap))
          t = t.mapGlobals('global.castRename(t.typ.globalType.insertFields(FastSeq((colsFieldName, newColsType)))))
        }

        if (entryMap.nonEmpty) {
          val newEntriesType = child.typ.entryArrayType.copy(elementType = child.typ.entryType.rename(entryMap))
          t = t.mapRows('row.castRename(t.typ.rowType.insertFields(FastSeq((entriesFieldName, newEntriesType)))))
        }

        t

      case MatrixKeyRowsBy(child, keys, isSorted) =>
        lower(child, ab).keyBy(keys, isSorted)

      case MatrixFilterRows(child, pred) =>
        lower(child, ab)
          .rename(Map(entriesFieldName -> MatrixType.entriesIdentifier))
          .filter(let(va = 'row,
            global = 'global.dropFields(colsField))
            in pred)
          .rename(Map(MatrixType.entriesIdentifier -> entriesFieldName))

      case MatrixFilterCols(child, pred) =>
        lower(child, ab)
          .mapGlobals('global.insertFields('newColIdx ->
            irRange(0, 'global (colsField).len)
              .filter('i ~>
                (let(sa = 'global (colsField)('i),
                  global = 'global.dropFields(colsField))
                  in pred))))
          .mapRows('row.insertFields(entriesField -> 'global ('newColIdx).map('i ~> 'row (entriesField)('i))))
          .mapGlobals('global
            .insertFields(colsField ->
              'global ('newColIdx).map('i ~> 'global (colsField)('i)))
            .dropFields('newColIdx))

      case MatrixAnnotateRowsTable(child, table, root, product) =>
        val kt = table.typ.keyType
        if (kt.size == 1 && kt.types(0) == TInterval(child.typ.rowKeyStruct.types(0)))
          TableIntervalJoin(lower(child, ab), lower(table, ab), root, product)
        else
          TableLeftJoinRightDistinct(lower(child, ab), lower(table, ab), root)

      case MatrixChooseCols(child, oldIndices) =>
        lower(child, ab)
          .mapGlobals('global.insertFields('newColIdx -> oldIndices.map(I32)))
          .mapRows('row.insertFields(entriesField -> 'global ('newColIdx).map('i ~> 'row (entriesField)('i))))
          .mapGlobals('global
            .insertFields(colsField -> 'global ('newColIdx).map('i ~> 'global (colsField)('i)))
            .dropFields('newColIdx))

      case MatrixAnnotateColsTable(child, table, root) =>
        val col = Symbol(genUID())
        val colKey = makeStruct(table.typ.key.zip(child.typ.colKey).map { case (tk, mck) => Symbol(tk) -> col(Symbol(mck)) }: _*)
        lower(child, ab)
          .mapGlobals(let(__dictfield = lower(table, ab)
            .keyBy(FastIndexedSeq())
            .collect()
            .apply('rows)
            .arrayStructToDict(table.typ.key)) {
            'global.insertFields(colsField ->
              'global (colsField).map(col ~> col.insertFields(Symbol(root) -> '__dictfield.invoke("get", colKey))))
          })

      case MatrixMapGlobals(child, newGlobals) =>
        lower(child, ab)
          .mapGlobals(
            let(global = 'global.dropFields(colsField)) {
              newGlobals
            }
              .insertFields(colsField -> 'global (colsField)))

      case MatrixMapRows(child, newRow) => {
        def liftScans(ir: IR): IRProxy = {
          val scans = new ArrayBuilder[(String, IR)]

          def f(ir: IR): IR = ir match {
            case x if IsScanResult(x) =>
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

        val lc = lower(child, ab)
        val e = Env[IR]("va" -> Ref("row", lc.typ.rowType),
          "global" -> SelectFields(Ref("global", lc.typ.globalType), child.typ.globalType.fieldNames))
        lc.mapRows(
          liftScans(Subst(newRow, BindingEnv(e, scan = Some(e), agg = Some(e))))
            .insertFields(entriesField -> 'row (entriesField)))
      }

      case MatrixMapCols(child, newCol, _) =>
        val loweredChild = lower(child, ab)

        val aggBuilder = new ArrayBuilder[(String, IR)]
        val scanBuilder = new ArrayBuilder[(String, IR)]

        def lift(ir: IR): IR = ir match {
          case x if IsScanResult(x) =>
            assert(!ContainsAgg(x))
            val s = genUID()
            scanBuilder += (s -> x)
            Ref(s, x.typ)
          case x if IsAggResult(x) =>
            assert(!ContainsScan(x))
            val s = genUID()
            aggBuilder += (s -> x)
            Ref(s, x.typ)
          case _ =>
            MapIR(lift)(ir)
        }

        val e = Env[IR]("global" -> SelectFields(Ref("global", loweredChild.typ.globalType),
          child.typ.globalType.fieldNames))
        val substEnv = BindingEnv(e, Some(e), Some(e))

        var b0 = lift(Subst(newCol, substEnv))
        val aggs = aggBuilder.result()
        val scans = scanBuilder.result()

        val idx = Ref(genUID(), TInt32())
        val idxSym = Symbol(idx.name)

        val aggTransformer: IRProxy => IRProxy = if (aggs.isEmpty)
          identity
        else {
          val aggStruct = MakeStruct(aggs)
          val aggResultArray = loweredChild.aggregate(
            aggLet(va = 'row) {
              irRange(0, 'global (colsField).len)
                .aggElements('__element_idx, '__result_idx, Some('global (colsField).len))(
                  let(sa = 'global (colsField)('__result_idx)) {
                    aggLet(sa = 'global (colsField)('__element_idx),
                      g = 'row (entriesField)('__element_idx)) {
                      aggFilter(!'g.isNA, aggStruct)
                    }
                  })
            })
          val ident = genUID()
          ab += ((ident, aggResultArray))

          val aggResultRef = Ref(genUID(), aggResultArray.typ)
          val aggResultElementRef = Ref(genUID(), aggResultArray.typ.asInstanceOf[TArray].elementType)

          b0 = aggs.foldLeft[IR](b0) { case (acc, (name, _)) => Let(name, GetField(aggResultElementRef, name), acc) }
          b0 = Let(aggResultElementRef.name, ArrayRef(aggResultRef, idx), b0)

          x: IRProxy => let.applyDynamicNamed("apply")((aggResultRef.name, RelationalRef(ident, aggResultArray.typ))).apply(x)
        }

        val scanTransformer: IRProxy => IRProxy = if (scans.isEmpty)
          identity
        else {
          val scanStruct = MakeStruct(scans)
          val scanResultArray = ArrayAggScan(
            GetField(Ref("global", loweredChild.typ.globalType), colsFieldName),
            "sa",
            scanStruct)

          val scanResultRef = Ref(genUID(), scanResultArray.typ)
          val scanResultElementRef = Ref(genUID(), scanResultArray.typ.asInstanceOf[TArray].elementType)

          b0 = scans.foldLeft[IR](b0) { case (acc, (name, _)) => Let(name, GetField(scanResultElementRef, name), acc) }
          b0 = Let(scanResultElementRef.name, ArrayRef(scanResultRef, idx), b0)

          x: IRProxy => let.applyDynamicNamed("apply")((scanResultRef.name, scanResultArray)).apply(x)
        }

        loweredChild.mapGlobals('global.insertFields(colsField -> scanTransformer(aggTransformer(
          irRange(0, 'global (colsField).len).map(idxSym ~> let(__cols_array = 'global (colsField), sa = '__cols_array (idxSym)) {
            b0
          })
        ))))

      case MatrixFilterEntries(child, pred) =>
        lower(child, ab).mapRows('row.insertFields(entriesField ->
          irRange(0, 'global (colsField).len).map {
            'i ~>
              let(g = 'row (entriesField)('i)) {
                irIf(let(sa = 'global (colsField)('i),
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
          lower(left, ab),
          lower(right, ab)
            .mapRows('row
              .insertFields(Symbol(rightEntries) -> 'row (entriesField))
              .selectFields(right.typ.rowKey :+ rightEntries: _*))
            .mapGlobals('global
              .insertFields(Symbol(rightCols) -> 'global (colsField))
              .selectFields(rightCols)),
          "inner")
          .mapRows('row
            .insertFields(entriesField ->
              makeArray('row (entriesField), 'row (Symbol(rightEntries))).flatMap('a ~> 'a))
            // TableJoin puts keys first; drop rightEntries, but also restore left row field order
            .selectFields(left.typ.rvRowType.fieldNames: _*))
          .mapGlobals('global
            .insertFields(colsField ->
              makeArray('global (colsField), 'global (Symbol(rightCols))).flatMap('a ~> 'a))
            .dropFields(Symbol(rightCols)))

      case MatrixMapEntries(child, newEntries) =>
        lower(child, ab).mapRows('row.insertFields(entriesField ->
          irRange(0, 'global (colsField).len).map {
            'i ~>
              let(g = 'row (entriesField)('i),
                sa = 'global (colsField)('i),
                va = 'row,
                global = 'global.dropFields(colsField)) {
                newEntries
              }
          }))

      case MatrixRepartition(child, n, shuffle) => TableRepartition(lower(child, ab), n, shuffle)

      case MatrixFilterIntervals(child, intervals, keep) => TableFilterIntervals(lower(child, ab), intervals, keep)

      case MatrixUnionRows(children) =>
        // FIXME: this should check that all children have the same column keys.
        TableUnion(MatrixUnionRows.unify(children).map(lower(_, ab)))

      case MatrixDistinctByRow(child) => TableDistinct(lower(child, ab))

      case MatrixRowsHead(child, n) => TableHead(lower(child, ab), n)

      case MatrixColsHead(child, n) => lower(child, ab)
        .mapGlobals('global.insertFields(colsField -> 'global (colsField).invoke("[:*]", n)))
        .mapRows('row.insertFields(entriesField -> 'row (entriesField).invoke("[:*]", n)))

      case MatrixExplodeCols(child, path) =>
        val loweredChild = lower(child, ab)
        val lengths = Symbol(genUID())
        val colIdx = Symbol(genUID())
        val nestedIdx = Symbol(genUID())
        val colElementUID1 = Symbol(genUID())


        val nestedRefs = path.init.scanLeft('global (colsField)(colIdx): IRProxy)((irp, name) => irp(Symbol(name)))
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
          .mapGlobals('global.insertFields(lengths -> 'global (colsField).map({
            colElementUID1 ~> arrayIR.len.orElse(0)
          })))
          .mapGlobals('global.insertFields(colsField ->
            irRange(0, 'global (colsField).len, 1)
              .flatMap({
                colIdx ~>
                  irRange(0, 'global (lengths)(colIdx), 1)
                    .map({
                      nestedIdx ~> postExplodeSelector
                    })
              })))
          .mapRows('row.insertFields(entriesField ->
            irRange(0, 'row (entriesField).len, 1)
              .flatMap(colIdx ~>
                irRange(0, 'global (lengths)(colIdx), 1).map(Symbol(genUID()) ~> 'row (entriesField)(colIdx)))))
          .mapGlobals('global.dropFields(lengths))

      case MatrixAggregateRowsByKey(child, entryExpr, rowExpr) =>

        val substEnv = BindingEnv[IRProxy](
          Env(("global", 'global.dropFields(colsField))),
          agg = Some(Env(("va", 'row), ("global", 'global.dropFields(colsField)))))
        val eeSub = subst(entryExpr, substEnv)
        val reSub = subst(rowExpr, substEnv)
        lower(child, ab)
          .aggregateByKey(
            reSub.insertFields(entriesField -> irRange(0, 'global (colsField).len)
              .aggElements('__element_idx, '__result_idx, Some('global (colsField).len))(
                let(sa = 'global (colsField)('__result_idx)) {
                  aggLet(sa = 'global (colsField)('__element_idx),
                    g = 'row (entriesField)('__element_idx)) {
                    eeSub
                  }
                })))

      case MatrixCollectColsByKey(child) =>
        lower(child, ab)
          .mapGlobals('global.insertFields('newColIdx ->
            irRange(0, 'global (colsField).len).map {
              'i ~>
                makeTuple('global (colsField)('i).selectFields(child.typ.colKey: _*),
                  'i)
            }.groupByKey.toArray))
          .mapRows('row.insertFields(entriesField ->
            'global ('newColIdx).map {
              'kv ~>
                makeStruct(child.typ.entryType.fieldNames.map { s =>
                  (Symbol(s), 'kv ('value).map {
                    'i ~> 'row (entriesField)('i)(Symbol(s))
                  })
                }: _*)
            }))
          .mapGlobals('global
            .insertFields(colsField ->
              'global ('newColIdx).map {
                'kv ~>
                  'kv ('key).insertFields(
                    child.typ.colValueStruct.fieldNames.map { s =>
                      (Symbol(s), 'kv ('value).map('i ~> 'global (colsField)('i)(Symbol(s))))
                    }: _*)
              })
            .dropFields('newColIdx)
          )

      case MatrixExplodeRows(child, path) => TableExplode(lower(child, ab), path)

      case mr: MatrixRead => mr.lower()

      case MatrixAggregateColsByKey(child, entryExpr, colExpr) =>
        val colKey = child.typ.colKey

        val originalColIdx = Symbol(genUID())
        val newColIdx1 = Symbol(genUID())
        val newColIdx2 = Symbol(genUID())
        val colsAggIdx = Symbol(genUID())
        val keyMap = Symbol(genUID())
        val aggElementIdx = Symbol(genUID())

        val substEnv = BindingEnv[IRProxy](
          Env(("global", 'global.dropFields(colsField, keyMap))),
          agg = Some(Env(("global", 'global.dropFields(colsField, keyMap)))))
        val ceSub = subst(colExpr, substEnv)
        val eeSub = subst(entryExpr, substEnv.bindEval("va", 'row).bindAgg("va", 'row))

        lower(child, ab)
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
            let(__entries = 'row (entriesField), __key_map = 'global (keyMap)) {
              irRange(0, '__key_map.len)
                .map(newColIdx1 ~> '__key_map (newColIdx1)
                  .apply('value)
                  .arrayAgg(aggElementIdx ~>
                    aggLet(g = '__entries (aggElementIdx), sa = 'global (colsField)(aggElementIdx)) {
                      eeSub
                    }))
            }))
          .mapGlobals(
            'global.insertFields(colsField ->
              let(__key_map = 'global (keyMap)) {
                irRange(0, '__key_map.len)
                  .map(newColIdx2 ~>
                    concatStructs(
                      '__key_map (newColIdx2)('key),
                      '__key_map (newColIdx2)('value)
                        .arrayAgg(colsAggIdx ~> aggLet(sa = 'global (colsField)(colsAggIdx)) {
                          ceSub
                        })
                    ))
              }
            ).dropFields(keyMap))

      case MatrixLiteral(value) => TableLiteral(value.toTableValue(colsFieldName, entriesFieldName))
    }

    assert(lowered.typ == loweredType(mir.typ),
      s"\n  ACTUAL: ${ lowered.typ }\n  EXPECT: ${ loweredType(mir.typ) }" +
        s"\n  BEFORE: ${ Pretty(mir) }\n  AFTER: ${ Pretty(lowered) }")
    lowered
  }


  private[this] def lower(tir: TableIR, ab: ArrayBuilder[(String, IR)]): TableIR = {
    val lowered = tir match {
      case CastMatrixToTable(child, entries, cols) =>
        TableRename(lower(child, ab), Map(entriesFieldName -> entries), Map(colsFieldName -> cols))

      case MatrixEntriesTable(child) =>
        val oldColIdx = Symbol(genUID())
        val lambdaIdx1 = Symbol(genUID())
        val currentColIdx = Symbol(genUID())
        lower(child, ab)
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
        val loweredChild = lower(child, ab)
        TableToTableApply(loweredChild,
          function.lower()
            .getOrElse(WrappedMatrixToTableFunction(function, colsFieldName, entriesFieldName, child.typ.colKey)))

      case MatrixRowsTable(child) =>
        lower(child, ab)
          .mapGlobals('global.dropFields(colsField))
          .mapRows('row.dropFields(entriesField))

      case MatrixColsTable(child) =>
        val colKey = child.typ.colKey
        let(__cols_and_globals = lower(child, ab).getGlobals) {
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

      case table => lowerChildren(table, ab).asInstanceOf[TableIR]
    }

    if (lowered.typ != tir.typ)
      fatal(s"lowering changed type:\n  before: ${ tir.typ }\n  after: ${ lowered.typ }")
    lowered
  }

  private[this] def lower(ir: IR, ab: ArrayBuilder[(String, IR)]): IR = {
    val lowered = ir match {
      case MatrixToValueApply(child, function) => TableToValueApply(lower(child, ab), function.lower()
        .getOrElse(WrappedMatrixToValueFunction(function, colsFieldName, entriesFieldName, child.typ.colKey)))
      case MatrixWrite(child, writer) =>
        TableWrite(lower(child, ab), WrappedMatrixWriter(writer, colsFieldName, entriesFieldName, child.typ.colKey))
      case MatrixMultiWrite(children, writer) =>
        TableMultiWrite(children.map(lower(_, ab)), WrappedMatrixNativeMultiWriter(writer, children.head.typ.colKey))
      case MatrixAggregate(child, query) =>
        val idx = Symbol(genUID())
        lower(child, ab)
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
      case _ => lowerChildren(ir, ab).asInstanceOf[IR]
    }
    if (lowered.typ != ir.typ)
      fatal(s"lowering changed type:\n  before: ${ ir.typ }\n  after: ${ lowered.typ }")
    lowered
  }
}

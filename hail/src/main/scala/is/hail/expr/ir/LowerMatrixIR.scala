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
      case bmir: BlockMatrixIR => lower(bmir, ab)
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

  import is.hail.expr.ir.IRBuilder._

  def matrixSubstEnv(child: MatrixIR): BindingEnv[IRProxy] = {
    val e = Env[IRProxy]("global" -> 'global.selectFields(child.typ.globalType.fieldNames: _*),
      "va" -> 'row.selectFields(child.typ.rowType.fieldNames: _*))
    BindingEnv(e, agg = Some(e), scan = Some(e))
  }

  def matrixSubstEnvIR(child: MatrixIR, lowered: TableIR): BindingEnv[IR] = {
    val e = Env[IR]("global" -> SelectFields(Ref("global", lowered.typ.globalType), child.typ.globalType.fieldNames),
      "va" -> SelectFields(Ref("row", lowered.typ.rowType), child.typ.rowType.fieldNames))
    BindingEnv(e, agg = Some(e), scan = Some(e))
  }


  private[this] def lower(mir: MatrixIR, ab: ArrayBuilder[(String, IR)]): TableIR = {
    val lowered = mir match {
      case RelationalLetMatrixTable(name, value, body) =>
        RelationalLetTable(name, lower(value, ab), lower(body, ab))

      case CastTableToMatrix(child, entries, cols, colKey) =>
        val lc = lower(child, ab)
        lc.mapRows(
          irIf('row (Symbol(entries)).isNA) {
            irDie("missing entry array unsupported in 'to_matrix_table_row_major'", lc.typ.rowType)
          } {
            irIf('row (Symbol(entries)).len.cne( 'global (Symbol(cols)).len)) {
              irDie("length mismatch between entry array and column array in 'to_matrix_table_row_major'", lc.typ.rowType)
            } {
              'row
            }
          }
        ).rename(Map(entries -> entriesFieldName), Map(cols -> colsFieldName))

      case MatrixToMatrixApply(child, function) =>
        val loweredChild = lower(child, ab)
        TableToTableApply(loweredChild, function.lower()
          .getOrElse(WrappedMatrixToMatrixFunction(function,
            colsFieldName,
            entriesFieldName,
            child.typ.colKey)))

      case MatrixRename(child, globalMap, colMap, rowMap, entryMap) =>
        var t = lower(child, ab).rename(rowMap, globalMap)

        if (colMap.nonEmpty) {
          val newColsType = TArray(child.typ.colType.rename(colMap))
          t = t.mapGlobals('global.castRename(t.typ.globalType.insertFields(FastSeq((colsFieldName, newColsType)))))
        }

        if (entryMap.nonEmpty) {
          val newEntriesType = TArray(child.typ.entryType.rename(entryMap))
          t = t.mapRows('row.castRename(t.typ.rowType.insertFields(FastSeq((entriesFieldName, newEntriesType)))))
        }

        t

      case MatrixKeyRowsBy(child, keys, isSorted) =>
        lower(child, ab).keyBy(keys, isSorted)

      case MatrixFilterRows(child, pred) =>
        lower(child, ab)
          .filter(subst(pred, matrixSubstEnv(child)))

      case MatrixFilterCols(child, pred) =>
        lower(child, ab)
          .mapGlobals('global.insertFields('newColIdx ->
            irRange(0, 'global (colsField).len)
              .filter('i ~>
                (let(sa = 'global (colsField)('i))
                  in subst(pred, matrixSubstEnv(child))))))
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
              'global (colsField).map(col ~> col.insertFields(Symbol(root) -> '__dictfield.invoke("get", table.typ.valueType, colKey))))
          })

      case MatrixMapGlobals(child, newGlobals) =>
        lower(child, ab)
          .mapGlobals(
            subst(newGlobals, BindingEnv(Env[IRProxy](
              "global" -> 'global.selectFields(child.typ.globalType.fieldNames: _*))))
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
        lc.mapRows(
          liftScans(Subst(newRow, matrixSubstEnvIR(child, lc)))
            .insertFields(entriesField -> 'row (entriesField)))
      }

      case MatrixMapCols(child, newCol, _) =>
        val loweredChild = lower(child, ab)

        val aggBuilder = new ArrayBuilder[(String, IR)]
        val scanBuilder = new ArrayBuilder[(String, IR)]
        val aggLetBuilder = new ArrayBuilder[(String, IR)]
        val scanLetBuilder = new ArrayBuilder[(String, IR)]

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
          case AggLet(name, value, body, isScan) =>
            val ab = if (isScan) scanLetBuilder else aggLetBuilder
            ab += ((name, value))
            lift(body)
          case _ =>
            MapIR(lift)(ir)
        }

        var b0 = lift(Subst(newCol, matrixSubstEnvIR(child,loweredChild)))
        val aggs = aggBuilder.result()
        val scans = scanBuilder.result()

        val idx = Ref(genUID(), TInt32())
        val idxSym = Symbol(idx.name)

        val aggTransformer: IRProxy => IRProxy = if (aggs.isEmpty)
          identity
        else {
          val aggStruct = aggLetBuilder.result().foldRight[IR](MakeStruct(aggs)) { case ((name, value), comb) =>
            AggLet(name, value, comb, isScan = false)
          }
          val aggResultArray = loweredChild.aggregate(
            aggLet(va = 'row.selectFields(child.typ.rowType.fieldNames: _*)) {
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
          val scanStruct = scanLetBuilder.result().foldRight[IR](MakeStruct(scans)) { case ((name, value), comb) =>
            AggLet(name, value, comb, isScan = true)
          }
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
        val lc = lower(child, ab)
          lc.mapRows('row.insertFields(entriesField ->
          irRange(0, 'global (colsField).len).map {
            'i ~>
              let(g = 'row (entriesField)('i)) {
                irIf(let(sa = 'global (colsField)('i))
                  in !subst(pred, matrixSubstEnv(child))) {
                  NA(child.typ.entryType)
                } {
                  'g
                }
              }
          }))

      case MatrixUnionCols(left, right) =>
        val rightEntries = genUID()
        val rightCols = genUID()
        val ll = lower(left, ab).distinct()
        TableJoin(
          ll,
          lower(right, ab).distinct()
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
            .selectFields(ll.typ.rowType.fieldNames: _*))
          .mapGlobals('global
            .insertFields(colsField ->
              makeArray('global (colsField), 'global (Symbol(rightCols))).flatMap('a ~> 'a))
            .dropFields(Symbol(rightCols)))

      case MatrixMapEntries(child, newEntries) =>
        lower(child, ab).mapRows('row.insertFields(entriesField ->
          irRange(0, 'global (colsField).len).map {
            'i ~>
              let(g = 'row (entriesField)('i),
                sa = 'global (colsField)('i)) {
                subst(newEntries, BindingEnv(Env(
                  "global" -> 'global.selectFields(child.typ.globalType.fieldNames: _*),
                  "va" -> 'row.selectFields(child.typ.rowType.fieldNames: _*))))
              }
          }))

      case MatrixRepartition(child, n, shuffle) => TableRepartition(lower(child, ab), n, shuffle)

      case MatrixFilterIntervals(child, intervals, keep) => TableFilterIntervals(lower(child, ab), intervals, keep)

      case MatrixUnionRows(children) =>
        // FIXME: this should check that all children have the same column keys.
        val first = lower(children.head, ab)
        TableUnion(FastIndexedSeq(first) ++
          children.tail.map(lower(_, ab)
            .mapRows('row.selectFields(first.typ.rowType.fieldNames: _*))))

      case MatrixDistinctByRow(child) => TableDistinct(lower(child, ab))

      case MatrixRowsHead(child, n) => TableHead(lower(child, ab), n)

      case MatrixColsHead(child, n) => lower(child, ab)
        .mapGlobals('global.insertFields(colsField -> 'global (colsField).invoke("[:*]", TArray(child.typ.colType), n)))
        .mapRows('row.insertFields(entriesField -> 'row (entriesField).invoke("[:*]", TArray(child.typ.entryType), n)))

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

        val substEnv = matrixSubstEnv(child)
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

        val substEnv = matrixSubstEnv(child)
        val ceSub = subst(colExpr, substEnv)
        val vaBinding = 'row.selectFields(child.typ.rowType.fieldNames: _*)
        val eeSub = subst(entryExpr, substEnv.bindEval("va", vaBinding).bindAgg("va", vaBinding))

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

      case MatrixLiteral(_, tl) => tl
    }

    if (!mir.typ.isCompatibleWith(lowered.typ))
      throw new RuntimeException(s"Lowering changed type:\n  BEFORE: ${ Pretty(mir) }\n    ${ mir.typ }\n    ${ mir.typ.canonicalTableType}\n  AFTER: ${ Pretty(lowered) }\n    ${ lowered.typ }")
    lowered
  }


  private[this] def lower(tir: TableIR, ab: ArrayBuilder[(String, IR)]): TableIR = {
    val lowered = tir match {
      case CastMatrixToTable(child, entries, cols) =>
        lower(child, ab)
          .mapRows('row.selectFields(child.typ.rowType.fieldNames ++ Array(entriesFieldName): _*))
          .rename(Map(entriesFieldName -> entries), Map(colsFieldName -> cols))

      case x@MatrixEntriesTable(child) =>
        val lc = lower(child, ab)

        if (child.typ.rowKey.nonEmpty && child.typ.colKey.nonEmpty) {
          val oldColIdx = Symbol(genUID())
          val lambdaIdx1 = Symbol(genUID())
          val lambdaIdx2 = Symbol(genUID())
          val lambdaIdx3 = Symbol(genUID())
          val toExplode = Symbol(genUID())
          val values = Symbol(genUID())
          lc
            .mapGlobals('global.insertFields(oldColIdx ->
              irRange(0, 'global (colsField).len)
                .map(lambdaIdx1 ~> makeStruct('key -> 'global (colsField)(lambdaIdx1).selectFields(child.typ.colKey: _*), 'value -> lambdaIdx1))
                .sort(ascending = true, onKey = true)
                .map(lambdaIdx1 ~> lambdaIdx1('value))))
            .aggregateByKey(makeStruct(values -> applyAggOp(Collect(), seqOpArgs = FastIndexedSeq('row.selectFields(lc.typ.valueType.fieldNames: _*)))))
            .mapRows('row.dropFields(values).insertFields(toExplode ->
              'global (oldColIdx)
                .flatMap(lambdaIdx1 ~> 'row (values)
                  .filter(lambdaIdx2 ~> !lambdaIdx2(entriesField)(lambdaIdx1).isNA)
                  .map(lambdaIdx3 ~> let(__col = 'global (colsField)(lambdaIdx1), __entry = lambdaIdx3(entriesField)(lambdaIdx1)) {
                    makeStruct(
                      child.typ.rowValueStruct.fieldNames.map(Symbol(_)).map(f => f -> lambdaIdx3(f)) ++
                        child.typ.colType.fieldNames.map(Symbol(_)).map(f => f -> '__col (f)) ++
                        child.typ.entryType.fieldNames.map(Symbol(_)).map(f => f -> '__entry (f)): _*
                    )
                  }))))

            .explode(toExplode)
            .mapRows(makeStruct(x.typ.rowType.fieldNames.map { f =>
              val fd = Symbol(f)
              (fd, if (child.typ.rowKey.contains(f)) 'row (fd) else 'row (toExplode) (fd)) }: _*))
            .mapGlobals('global.dropFields(colsField, oldColIdx))
            .keyBy(child.typ.rowKey ++ child.typ.colKey, isSorted = true)
        } else {
          val colIdx = Symbol(genUID())
          val lambdaIdx = Symbol(genUID())
          val result = lc
            .mapRows('row.insertFields(colIdx -> irRange(0, 'global (colsField).len)
              .filter(lambdaIdx ~> !'row (entriesField)(lambdaIdx).isNA)))
            .explode(colIdx)
            .mapRows(let(__col_struct = 'global (colsField)('row (colIdx)),
              __entry_struct = 'row (entriesField)('row (colIdx))) {
              val newFields = child.typ.colType.fieldNames.map(Symbol(_)).map(f => f -> '__col_struct (f)) ++
                child.typ.entryType.fieldNames.map(Symbol(_)).map(f => f -> '__entry_struct (f))

              'row.dropFields(entriesField, colIdx).insertFieldsList(newFields,
                ordering = Some(x.typ.rowType.fieldNames.toFastIndexedSeq))
            })
            .mapGlobals('global.dropFields(colsField))
          if (child.typ.colKey.isEmpty)
            result
          else {
            assert(child.typ.rowKey.isEmpty)
            result.keyBy(child.typ.colKey)
          }
        }

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

    assertTypeUnchanged(tir, lowered)
    lowered
  }

  private[this] def lower(bmir: BlockMatrixIR, ab: ArrayBuilder[(String, IR)]): BlockMatrixIR = {
    val lowered = bmir match {
      case noMatrixChildren => lowerChildren(noMatrixChildren, ab).asInstanceOf[BlockMatrixIR]
    }
    assertTypeUnchanged(bmir, lowered)
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
        val lc = lower(child, ab)
        val idx = Symbol(genUID())
        lc
          .aggregate(
            aggLet(
              __entries_field = 'row (entriesField),
              __cols_field = 'global (colsField)) {
              irRange(0, '__entries_field.len)
                .filter(idx ~> !'__entries_field (idx).isNA)
                .aggExplode(idx ~> aggLet(sa = '__cols_field (idx), g = '__entries_field (idx)) {
                  subst(query, matrixSubstEnv(child))
                })
            })
      case _ => lowerChildren(ir, ab).asInstanceOf[IR]
    }
    assertTypeUnchanged(ir, lowered)
    lowered
  }

  private[this] def assertTypeUnchanged(original: BaseIR, lowered: BaseIR) {
    if (lowered.typ != original.typ)
      fatal(s"lowering changed type:\n  before: ${ original.typ }\n after: ${ lowered.typ }\n")
  }
}

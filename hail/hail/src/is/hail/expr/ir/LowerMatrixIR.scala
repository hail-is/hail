package is.hail.expr.ir

import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.collection.compat.mutable.Growable
import is.hail.expr.ir.{Memoized => M}
import is.hail.expr.ir.defs._
import is.hail.expr.ir.functions.{WrappedMatrixToTableFunction, WrappedMatrixToValueFunction}
import is.hail.types.virtual._
import is.hail.utils._

object LowerMatrixIR {
  val entriesFieldName: String = MatrixType.entriesIdentifier
  val colsFieldName: String = "__cols"
  val colsField: Symbol = Symbol(colsFieldName)
  val entriesField: Symbol = Symbol(entriesFieldName)

  def apply(ctx: ExecuteContext, ir0: BaseIR): BaseIR = {
    val ab = ArraySeq.newBuilder[(Name, IR)]

    val lowered =
      ir0 match {
        case ir: IR =>
          val l1 = lower(ctx, ir, ab)
          ab.result().foldRight[IR](l1) { case ((ident, value), body) =>
            RelationalLet(ident, value, body)
          }
        case tir: TableIR =>
          val l1 = lower(ctx, tir, ab)
          ab.result().foldRight[TableIR](l1) { case ((ident, value), body) =>
            RelationalLetTable(ident, value, body)
          }
        case mir: MatrixIR =>
          val l1 = lower(ctx, mir, ab)
          ab.result().foldRight[TableIR](l1) { case ((ident, value), body) =>
            RelationalLetTable(ident, value, body)
          }
        case bmir: BlockMatrixIR =>
          val l1 = lower(ctx, bmir, ab)
          assert(ab.result().isEmpty)
          l1
      }

    NormalizeNames()(ctx, lowered)
  }

  private def lowerChildren(
    ctx: ExecuteContext,
    ir: BaseIR,
    ab: Growable[(Name, IR)],
  ): BaseIR =
    ir.mapChildren {
      case tir: TableIR => lower(ctx, tir, ab)
      case mir: MatrixIR => throw new RuntimeException(s"expect specialized lowering rule for " +
          s"${ir.getClass.getName}\n  Found MatrixIR child $mir")
      case bmir: BlockMatrixIR => lower(ctx, bmir, ab)
      case vir: IR => lower(ctx, vir, ab)
    }

  def colVals(tir: TableIR): IR =
    GetField(Ref(TableIR.globalName, tir.typ.globalType), colsFieldName)

  def globals(tir: TableIR): IR = {
    val globalType = tir.typ.globalType
    SelectFields(
      Ref(TableIR.globalName, globalType),
      globalType.fieldNames.diff(FastSeq(colsFieldName)),
    )
  }

  def rowVal(tir: TableIR): IR = {
    val rowType = tir.typ.rowType
    SelectFields(
      Ref(TableIR.rowName, rowType),
      rowType.fieldNames.diff(FastSeq(entriesFieldName)),
    )
  }

  def entries(tir: TableIR): IR =
    GetField(Ref(TableIR.rowName, tir.typ.rowType), entriesFieldName)

  import is.hail.expr.ir.DeprecatedIRBuilder._

  private def bindingsToStruct(bindings: IndexedSeq[(Name, IR)]): MakeStruct =
    MakeStruct(bindings.map { case (n, ir) => n.str -> ir })

  private def unwrapStruct(bindings: IndexedSeq[(Name, _)], struct: Atom): IndexedSeq[(Name, IR)] =
    bindings.map { case (name, _) => name -> GetField(struct, name.str) }

  private def lower(
    ctx: ExecuteContext,
    mir: MatrixIR,
    liftedRelationalLets: Growable[(Name, IR)],
  ): TableIR = {
    val lowered = mir match {
      case RelationalLetMatrixTable(name, value, body) =>
        RelationalLetTable(
          name,
          lower(ctx, value, liftedRelationalLets),
          lower(ctx, body, liftedRelationalLets),
        )

      case CastTableToMatrix(child, entries, cols, _) =>
        val lc = lower(ctx, child, liftedRelationalLets)
        val row: Atom = Ref(TableIR.rowName, lc.typ.rowType)
        val glob = Ref(TableIR.globalName, lc.typ.globalType)
        TableMapRows(
          lc,
          bindIR(GetField(row, entries)) { entries =>
            If(
              IsNA(entries),
              Die("missing entry array unsupported in 'to_matrix_table_row_major'", row.typ),
              bindIRs(ArrayLen(entries), ArrayLen(GetField(glob, cols))) {
                case Seq(entriesLen, colsLen) =>
                  If(
                    entriesLen cne colsLen,
                    Die(
                      strConcat(
                        "length mismatch between entry array and column array in 'to_matrix_table_row_major': ",
                        entriesLen,
                        " entries, ",
                        colsLen,
                        " cols, at ",
                        SelectFields(row, child.typ.key),
                      ),
                      row.typ,
                      -1,
                    ),
                    row,
                  )
              },
            )
          },
        ).rename(Map(entries -> entriesFieldName), Map(cols -> colsFieldName))

      case MatrixToMatrixApply(child, function) =>
        val loweredChild = lower(ctx, child, liftedRelationalLets)
        TableToTableApply(loweredChild, function.lower())

      case MatrixRename(child, globalMap, colMap, rowMap, entryMap) =>
        var t = lower(ctx, child, liftedRelationalLets).rename(rowMap, globalMap)

        if (colMap.nonEmpty) {
          val newColsType = TArray(child.typ.colType.rename(colMap))
          t = t.mapGlobals('global.castRename(t.typ.globalType.insertFields(FastSeq(
            colsFieldName -> newColsType
          ))))
        }

        if (entryMap.nonEmpty) {
          val newEntriesType = TArray(child.typ.entryType.rename(entryMap))
          t = t.mapRows('row.castRename(t.typ.rowType.insertFields(FastSeq(
            entriesFieldName -> newEntriesType
          ))))
        }

        t

      case MatrixKeyRowsBy(child, keys, isSorted) =>
        lower(ctx, child, liftedRelationalLets).keyBy(keys, isSorted)

      case MatrixFilterRows(child, pred) =>
        lower(ctx, child, liftedRelationalLets)
          .filter(
            let(
              global = 'global.selectFields(child.typ.globalType.fieldNames: _*),
              va = 'row.selectFields(child.typ.rowType.fieldNames: _*),
            ) in lower(ctx, pred, liftedRelationalLets)
          )

      case MatrixFilterCols(child, pred) =>
        lower(ctx, child, liftedRelationalLets)
          .mapGlobals(
            'global.insertFields(
              '__new_col_idx ->
                (let(
                  __cols = 'global(colsField),
                  global = 'global.selectFields(child.typ.globalType.fieldNames: _*),
                ) in irRange(0, '__cols.len).filter('__col_idx ~>
                  (let(sa = '__cols('__col_idx)) in
                    lower(ctx, pred, liftedRelationalLets))))
            )
          )
          .mapRows(
            let(__entries = 'row(entriesField)) in
              'row.insertFields(entriesField -> 'global('__new_col_idx).map('i ~> '__entries('i)))
          )
          .mapGlobals(
            let(__cols = 'global(colsField)) in
              'global
                .insertFields(colsField -> 'global('__new_col_idx).map('i ~> '__cols('i)))
                .dropFields('__new_col_idx)
          )

      case MatrixAnnotateRowsTable(child, table, root, product) =>
        val kt = table.typ.keyType
        if (kt.size == 1 && kt.types(0) == TInterval(child.typ.rowKeyStruct.types(0)))
          TableIntervalJoin(
            lower(ctx, child, liftedRelationalLets),
            lower(ctx, table, liftedRelationalLets),
            root,
            product,
          )
        else
          TableLeftJoinRightDistinct(
            lower(ctx, child, liftedRelationalLets),
            lower(ctx, table, liftedRelationalLets),
            root,
          )

      case MatrixChooseCols(child, oldIndices) =>
        lower(ctx, child, liftedRelationalLets)
          .mapGlobals('global.insertFields('__new_col_idx -> Literal(TArray(TInt32), oldIndices)))
          .mapRows(
            let(__entries = 'row(entriesField)) in
              'row.insertFields(entriesField -> 'global('__new_col_idx).map('i ~> '__entries('i)))
          )
          .mapGlobals(
            let(__cols = 'global(colsField)) in
              'global
                .insertFields(colsField -> 'global('__new_col_idx).map('i ~> '__cols('i)))
                .dropFields('__new_col_idx)
          )

      case MatrixAnnotateColsTable(child, table, root) =>
        lower(ctx, child, liftedRelationalLets)
          .mapGlobals(
            let(
              __dictfield =
                lower(ctx, table, liftedRelationalLets)
                  .keyBy(FastSeq())
                  .collect()
                  .apply('rows)
                  .arrayStructToDict(table.typ.key)
            ) in 'global.insertFields(
              colsField -> {
                val key =
                  makeStruct(table.typ.key.zip(child.typ.colKey).map { case (tk, mck) =>
                    Symbol(tk) -> '__cols(Symbol(mck))
                  }: _*)

                'global(colsField).map('__cols ~>
                  '__cols.insertFields(
                    Symbol(root) -> '__dictfield.invoke("get", table.typ.valueType, key)
                  ))
              }
            )
          )

      case MatrixMapGlobals(child, newGlobals) =>
        lower(ctx, child, liftedRelationalLets)
          .mapGlobals(
            (let(global = 'global.selectFields(child.typ.globalType.fieldNames: _*)) in
              lower(ctx, newGlobals, liftedRelationalLets))
              .insertFields(colsField -> 'global(colsField))
          )

      case MatrixMapRows(child, newRow) =>
        def liftScans(ir: IR): IRProxy = {
          def lift(ir: IR, builder: Growable[(Name, IR)]): IR = ir match {
            case a: ApplyScanOp =>
              val s = freshName()
              builder += (s -> a)
              Ref(s, a.typ)

            case a @ AggFold(_, _, _, _, _, true) =>
              val s = freshName()
              builder += (s -> a)
              Ref(s, a.typ)

            case AggFilter(filt, body, true) =>
              val ab = ArraySeq.newBuilder[(Name, IR)]
              val liftedBody = lift(body, ab)
              val aggs = ab.result()
              val structResult = bindingsToStruct(aggs)
              val uid = Ref(freshName(), structResult.typ)
              builder += (uid.name -> AggFilter(filt, structResult, true))
              Let(unwrapStruct(aggs, uid), liftedBody)

            case AggExplode(a, name, body, true) =>
              val ab = ArraySeq.newBuilder[(Name, IR)]
              val liftedBody = lift(body, ab)
              val aggs = ab.result()
              val structResult = bindingsToStruct(aggs)
              val uid = Ref(freshName(), structResult.typ)
              builder += (uid.name -> AggExplode(a, name, structResult, true))
              Let(unwrapStruct(aggs, uid), liftedBody)

            case AggGroupBy(a, body, true) =>
              val ab = ArraySeq.newBuilder[(Name, IR)]
              val liftedBody = lift(body, ab)
              val aggs = ab.result()

              val aggIR = AggGroupBy(a, bindingsToStruct(aggs), true)
              val uid = Ref(freshName(), aggIR.typ)
              builder += (uid.name -> aggIR)

              ToDict(mapIR(ToStream(uid)) { eltUID =>
                bindIR(GetField(eltUID, "value")) { value =>
                  Let(
                    unwrapStruct(aggs, value),
                    maketuple(GetField(eltUID, "key"), liftedBody),
                  )
                }
              })

            case AggArrayPerElement(a, elementName, indexName, body, knownLength, true) =>
              val ab = ArraySeq.newBuilder[(Name, IR)]
              val liftedBody = lift(body, ab)

              val aggs = ab.result()
              val aggBody = bindingsToStruct(aggs)
              val aggIR = AggArrayPerElement(a, elementName, indexName, aggBody, knownLength, true)
              val uid = Ref(freshName(), aggIR.typ)
              builder += (uid.name -> aggIR)

              ToArray(mapIR(ToStream(uid)) { eltUID =>
                Let(unwrapStruct(aggs, eltUID), liftedBody)
              })

            case Block(bindings, body) =>
              val newBindings = ArraySeq.newBuilder[Binding]
              def go(i: Int, builder: Growable[(Name, IR)]): IR = {
                if (i == bindings.length) lift(body, builder)
                else bindings(i) match {
                  case Binding(name, value, Scope.SCAN) =>
                    val ab = ArraySeq.newBuilder[(Name, IR)]
                    val liftedBody = go(i + 1, ab)
                    val aggs = ab.result()
                    val structResult = bindingsToStruct(aggs)
                    val uid = Ref(freshName(), structResult.typ)
                    builder += (uid.name -> Let(FastSeq(name -> value), structResult))
                    newBindings ++= unwrapStruct(aggs, uid).map(b =>
                      Binding(b._1, b._2, Scope.EVAL)
                    )
                    liftedBody
                  case Binding(name, value, scope) =>
                    newBindings += Binding(name, lift(value, builder), scope)
                    go(i + 1, builder)
                }
              }
              val newBody = go(0, builder)
              Block(newBindings.result(), newBody)

            case _ =>
              MapIR(lift(_, builder))(ir)
          }

          val ab = ArraySeq.newBuilder[(Name, IR)]
          val b0 = lift(ir, ab)

          val b1 = if (ContainsAgg(b0))
            irRange(0, '__entries.len)
              .filter('i ~> !'__entries('i).isNA)
              .streamAgg('i ~> (aggLet(sa = '__cols('i), g = '__entries('i)) in b0))
          else
            irToProxy(b0)

          scanLet(
            global = 'global.selectFields(child.typ.globalType.fieldNames: _*),
            va = 'row.selectFields(child.typ.rowType.fieldNames: _*),
          ) in (letDyn(ab.result().map { case (name, expr) => name -> irToProxy(expr) }: _*) in b1)
        }

        lower(ctx, child, liftedRelationalLets).mapRows(
          (let(
            __cols = 'global(colsField),
            __entries = 'row(entriesField),
            n_cols = '__cols.len,
            global = 'global.selectFields(child.typ.globalType.fieldNames: _*),
            va = 'row.selectFields(child.typ.rowType.fieldNames: _*),
          ) in liftScans(lower(ctx, newRow, liftedRelationalLets)))
            .insertFields(entriesField -> 'row(entriesField))
        )

      case MatrixMapCols(child, newCol, _) =>
        val lc = lower(ctx, child, liftedRelationalLets)

        def lift(ir: IR, scanBindings: Growable[(Name, IR)], aggBindings: Growable[(Name, IR)])
          : IR = ir match {
          case a: ApplyScanOp =>
            val s = freshName()
            scanBindings += (s -> a)
            Ref(s, a.typ)

          case a: ApplyAggOp =>
            val s = freshName()
            aggBindings += (s -> a)
            Ref(s, a.typ)

          case a @ AggFold(_, _, _, _, _, isScan) =>
            val s = freshName()
            if (isScan) scanBindings += (s -> a)
            else aggBindings += (s -> a)
            Ref(s, a.typ)

          case AggFilter(filt, body, isScan) =>
            val ab = ArraySeq.newBuilder[(Name, IR)]
            val (liftedBody, builder) =
              if (isScan) (lift(body, ab, aggBindings), scanBindings)
              else (lift(body, scanBindings, ab), aggBindings)

            val aggs = ab.result()
            val structResult = bindingsToStruct(aggs)

            val uid = Ref(freshName(), structResult.typ)
            builder += (uid.name -> AggFilter(filt, structResult, isScan))
            Let(unwrapStruct(aggs, uid), liftedBody)

          case AggExplode(a, name, body, isScan) =>
            val ab = ArraySeq.newBuilder[(Name, IR)]
            val (liftedBody, builder) =
              if (isScan) (lift(body, ab, aggBindings), scanBindings)
              else (lift(body, scanBindings, ab), aggBindings)

            val aggs = ab.result()
            val structResult = bindingsToStruct(aggs)
            val uid = Ref(freshName(), structResult.typ)
            builder += (uid.name -> AggExplode(a, name, structResult, isScan))
            Let(unwrapStruct(aggs, uid), liftedBody)

          case AggGroupBy(a, body, isScan) =>
            val ab = ArraySeq.newBuilder[(Name, IR)]
            val (liftedBody, builder) =
              if (isScan) (lift(body, ab, aggBindings), scanBindings)
              else (lift(body, scanBindings, ab), aggBindings)

            val aggs = ab.result()
            val aggIR = AggGroupBy(a, bindingsToStruct(aggs), isScan)
            val uid = Ref(freshName(), aggIR.typ)
            builder += (uid.name -> aggIR)
            ToDict(mapIR(ToStream(uid)) { eltUID =>
              maketuple(
                GetField(eltUID, "key"),
                bindIR(GetField(eltUID, "value")) { value =>
                  Let(unwrapStruct(aggs, value), liftedBody)
                },
              )
            })

          case AggArrayPerElement(a, elementName, indexName, body, knownLength, isScan) =>
            val ab = ArraySeq.newBuilder[(Name, IR)]
            val (liftedBody, builder) =
              if (isScan) (lift(body, ab, aggBindings), scanBindings)
              else (lift(body, scanBindings, ab), aggBindings)

            val aggs = ab.result()
            val aggBody = bindingsToStruct(aggs)
            val aggIR =
              AggArrayPerElement(a, elementName, indexName, aggBody, knownLength, isScan)
            val uid = Ref(freshName(), aggIR.typ)
            builder += (uid.name -> aggIR)
            ToArray(mapIR(ToStream(uid))(eltUID => Let(unwrapStruct(aggs, eltUID), liftedBody)))

          case Block(bindings, body) =>
            val newBindings = ArraySeq.newBuilder[Binding]
            def go(i: Int, scanBindings: Growable[(Name, IR)], aggBindings: Growable[(Name, IR)])
              : IR =
              if (i == bindings.length) lift(body, scanBindings, aggBindings)
              else bindings(i) match {
                case Binding(name, value, Scope.EVAL) =>
                  val lifted = lift(value, scanBindings, aggBindings)
                  val liftedBody = go(i + 1, scanBindings, aggBindings)
                  newBindings += Binding(name, lifted, Scope.EVAL)
                  liftedBody
                case Binding(name, value, scope) =>
                  val ab = ArraySeq.newBuilder[(Name, IR)]
                  val liftedBody =
                    if (scope == Scope.SCAN) go(i + 1, ab, aggBindings)
                    else go(i + 1, scanBindings, ab)

                  val builder = if (scope == Scope.SCAN) scanBindings else aggBindings

                  val aggs = ab.result()
                  val structResult = bindingsToStruct(aggs)

                  val uid = Ref(freshName(), structResult.typ)
                  builder += (uid.name -> Block(FastSeq(Binding(name, value, scope)), structResult))
                  newBindings ++= unwrapStruct(aggs, uid).map(b =>
                    Binding(b._1, b._2, Scope.EVAL)
                  )
                  liftedBody
              }

            val newBody = go(0, scanBindings, aggBindings)
            Block(newBindings.result().reverse, newBody)

          case x: StreamAgg => x
          case x: StreamAggScan => x

          case _ =>
            MapIR(lift(_, scanBindings, aggBindings))(ir)
        }

        val scanBuilder = ArraySeq.newBuilder[(Name, IR)]
        val aggBuilder = ArraySeq.newBuilder[(Name, IR)]

        val b0 = lift(
          lower(ctx, newCol, liftedRelationalLets),
          scanBuilder,
          aggBuilder,
        )

        val aggs = aggBuilder.result()
        val scans = scanBuilder.result()

        val noOp: (IRProxy => IRProxy, IRProxy => IRProxy) =
          (identity[IRProxy], identity[IRProxy])

        val (
          aggOutsideTransformer: (IRProxy => IRProxy),
          aggInsideTransformer: (IRProxy => IRProxy),
        ) =
          if (aggs.isEmpty) noOp
          else {
            val aggResult =
              lc.deepCopy.aggregate(
                let(
                  __cols = 'global(colsField),
                  global = 'global.selectFields(child.typ.globalType.fieldNames: _*),
                ) in (aggLet(
                  __cols = 'global(colsField),
                  __entries = 'row(entriesField),
                  global = 'global.selectFields(child.typ.globalType.fieldNames: _*),
                  va = 'row.selectFields(child.typ.rowType.fieldNames: _*),
                ) in makeStruct(
                  'n_rows ->
                    applyAggOp(Count(), FastSeq(), FastSeq()),
                  'array_aggs ->
                    irRange(0, '__cols.len)
                      .aggElements('__element_idx, '__result_idx, Some('__cols.len))(
                        let(sa = '__cols('__result_idx)) in
                          (aggLet(sa = '__cols('__element_idx), g = '__entries('__element_idx)) in
                            aggFilter(!'g.isNA, bindingsToStruct(aggs)))
                      ),
                ))
              )

            val ident = freshName()
            liftedRelationalLets += (ident -> aggResult)

            val bindResult: IRProxy => IRProxy =
              let(
                __agg_result = RelationalRef(ident, aggResult.typ),
                __array_aggs = '__agg_result('array_aggs),
                n_rows = '__agg_result('n_rows),
              ) in _

            def bodyResult(body: IRProxy): IRProxy =
              let(__agg_elem = '__array_aggs('__col_idx)) in
                (letDyn(aggs.map { case (n, _) => n -> '__agg_elem(Symbol(n.str)) }: _*) in
                  body)

            (bindResult, bodyResult _)
          }

        val (
          scanOutsideTransformer: (IRProxy => IRProxy),
          scanInsideTransformer: (IRProxy => IRProxy),
        ) =
          if (scans.isEmpty) noOp
          else {
            val scanStruct = bindingsToStruct(scans)

            val bindResult: IRProxy => IRProxy =
              let(__scan_result = '__cols.streamAggScan('sa ~> scanStruct)) in _

            def bodyResult(body: IRProxy): IRProxy =
              let(__scan_elem = '__scan_result('__col_idx)) in
                (letDyn(scans.map { case (n, _) => n -> '__scan_elem(Symbol(n.str)) }: _*) in
                  body)

            (bindResult, bodyResult _)
          }

        lc.mapGlobals(
          let(
            __cols = 'global(colsField),
            global = 'global.selectFields(child.typ.globalType.fieldNames: _*),
          ) in 'global.insertFields(
            colsField ->
              aggOutsideTransformer(
                scanOutsideTransformer(
                  irRange(0, '__cols.len).map('__col_idx ~>
                    (let(sa = '__cols('__col_idx)) in
                      aggInsideTransformer(scanInsideTransformer(b0))))
                )
              )
          )
        )

      case MatrixFilterEntries(child, pred) =>
        val mtype = child.typ
        lower(ctx, child, liftedRelationalLets)
          .mapRows(
            let(
              __cols = 'global(colsField),
              __entries = 'row(entriesField),
              global = 'global.selectFields(child.typ.globalType.fieldNames: _*),
              va = 'row.selectFields(mtype.rowType.fieldNames: _*),
            ) in 'row.insertFields(
              entriesField ->
                irRange(0, '__cols.len).map('i ~>
                  (let(sa = '__cols('i), g = '__entries('i)) in
                    irIf(lower(ctx, pred, liftedRelationalLets))('g)(NA(mtype.entryType))))
            )
          )

      case MatrixUnionCols(left, right, joinType) =>
        def handleMissingEntriesArray(entries: Symbol, cols: Symbol): IRProxy =
          if (joinType == "inner") 'row(entries)
          else let(__entries = 'row(entries)) in
            irIf(!'__entries.isNA)('__entries)(
              irRange(0, 'global(cols).len).map('a ~>
                MakeStruct(right.typ.entryType.fields.map(f => (f.name, NA(f.typ)))))
            )

        val ll = lower(ctx, left, liftedRelationalLets).distinct()
        val rr = lower(ctx, right, liftedRelationalLets).distinct()
        TableJoin(
          ll,
          rr.mapRows(
            'row.castRename(rr.typ.rowType.rename(Map(entriesFieldName -> '__right_entries.name)))
          )
            .mapGlobals('global
              .insertFields('__right_cols -> 'global(colsField))
              .selectFields('__right_cols.name)),
          joinType,
        )
          .mapRows('row
            .insertFields(
              entriesField -> {
                val ls = handleMissingEntriesArray(entriesField, colsField)
                val rs = handleMissingEntriesArray('__right_entries, '__right_cols)
                makeArray(ls, rs).flatten
              }
            )
            .dropFields('__right_entries))
          .mapGlobals('global
            .insertFields(
              colsField ->
                makeArray('global(colsField), 'global('__right_cols)).flatten
            )
            .dropFields('__right_cols))

      case MatrixMapEntries(child, newEntries) =>
        val lc = lower(ctx, child, liftedRelationalLets)
        TableMapRows(
          lc,
          M.eval {
            for {
              cols <- Name("__cols") -> colVals(lc)
              entries <- Name("__entries") -> entries(lc)
              _ <- MatrixIR.globalName -> globals(lc)
              row <- MatrixIR.rowName -> rowVal(lc)
            } yield InsertFields(
              row,
              FastSeq(
                entriesFieldName ->
                  ToArray(StreamZip(
                    FastSeq(ToStream(cols), ToStream(entries)),
                    FastSeq(MatrixIR.colName, MatrixIR.entryName),
                    lower(ctx, newEntries, liftedRelationalLets),
                    ArrayZipBehavior.AssumeSameLength,
                  ))
              ),
            )
          },
        )

      case MatrixRepartition(child, n, shuffle) =>
        TableRepartition(lower(ctx, child, liftedRelationalLets), n, shuffle)

      case MatrixFilterIntervals(child, intervals, keep) =>
        TableFilterIntervals(lower(ctx, child, liftedRelationalLets), intervals, keep)

      case MatrixUnionRows(children) =>
        // FIXME: this should check that all children have the same column keys.
        val first = lower(ctx, children.head, liftedRelationalLets)
        TableUnion(FastSeq(first) ++
          children.tail.map(lower(ctx, _, liftedRelationalLets)
            .mapRows('row.selectFields(first.typ.rowType.fieldNames: _*))))

      case MatrixDistinctByRow(child) => TableDistinct(lower(ctx, child, liftedRelationalLets))

      case MatrixRowsHead(child, n) => TableHead(lower(ctx, child, liftedRelationalLets), n)
      case MatrixRowsTail(child, n) => TableTail(lower(ctx, child, liftedRelationalLets), n)

      case MatrixColsHead(child, n) =>
        lower(ctx, child, liftedRelationalLets)
          .mapGlobals('global.insertFields('__cols -> 'global('__cols).arraySlice(0, Some(n), 1)))
          .mapRows('row.insertFields(entriesField -> 'row(entriesField).arraySlice(0, Some(n), 1)))

      case MatrixColsTail(child, n) =>
        lower(ctx, child, liftedRelationalLets)
          .mapGlobals('global.insertFields('__cols -> 'global('__cols).arraySlice(-n, None, 1)))
          .mapRows('row.insertFields(entriesField -> 'row(entriesField).arraySlice(-n, None, 1)))

      case MatrixExplodeCols(child, path) =>
        lower(ctx, child, liftedRelationalLets)
          .mapGlobals(
            let(
              __cols =
                'global(colsField),
              __lengths =
                '__cols.map('__elem ~>
                  path
                    .foldLeft[IRProxy]('__elem) { case (irp, f) => irp(Symbol(f)) }
                    .len
                    .orElse(0)),
            ) in 'global.insertFields(
              '__cols ->
                irRange(0, '__cols.len).flatMap('__col_idx ~> {
                  val nestedRefs =
                    path.init.scanLeft('__cols('__col_idx))((irp, name) => irp(Symbol(name)))

                  irRange(0, '__lengths('__col_idx)).map('__length_idx ~>
                    path.zip(nestedRefs).zipWithIndex.foldRight[IRProxy]('__length_idx) {
                      case (((field, ref), i), arg) =>
                        val s = Symbol(field)
                        ref.insertFields(
                          s -> (if (i == nestedRefs.length - 1) ref(s).toArray(arg) else arg)
                        )
                    })
                }),
              '__lengths ->
                '__lengths,
            )
          )
          .mapRows(
            let(__entries = 'row(entriesField), __lengths = 'global('__lengths)) in
              'row.insertFields(
                entriesField ->
                  irRange(0, '__entries.len).flatMap('__col_idx ~>
                    irRange(0, '__lengths('__col_idx)).map('__unused ~>
                      '__entries('__col_idx)))
              )
          )
          .mapGlobals('global.dropFields('__lengths))

      case MatrixAggregateRowsByKey(child, entryExpr, rowExpr) =>
        lower(ctx, child, liftedRelationalLets)
          .aggregateByKey(
            let(
              __cols = 'global(colsField),
              global = 'global.selectFields(child.typ.globalType.fieldNames: _*),
            ) in (aggLet(
              __cols = 'global(colsField),
              __entries = 'row(entriesField),
              global = 'global.selectFields(child.typ.globalType.fieldNames: _*),
              va = 'row.selectFields(child.typ.rowType.fieldNames: _*),
            ) in lower(ctx, rowExpr, liftedRelationalLets).insertFields(
              entriesField ->
                irRange(0, '__cols.len)
                  .aggElements('__element_idx, '__result_idx, Some('__cols.len))(
                    let(sa = '__cols('__result_idx)) in
                      (aggLet(sa = '__cols('__element_idx), g = '__entries('__element_idx)) in
                        aggFilter(!'g.isNA, lower(ctx, entryExpr, liftedRelationalLets)))
                  )
            ))
          )

      case MatrixCollectColsByKey(child) =>
        lower(ctx, child, liftedRelationalLets)
          .mapGlobals(
            let(__cols = 'global(colsField)) in
              'global.insertFields(
                '__new_col_idx ->
                  irRange(0, '__cols.len)
                    .map('i ~> makeTuple('__cols('i).selectFields(child.typ.colKey: _*), 'i))
                    .groupByKey
                    .toArray
              )
          )
          .mapRows(
            let(__entries = 'row(entriesField)) in
              'row.insertFields(
                entriesField ->
                  'global('__new_col_idx).map {
                    'kv ~>
                      makeStruct(child.typ.entryType.fieldNames.map { f =>
                        val s = Symbol(f)
                        s -> 'kv('value).map('i ~> '__entries('i)(s))
                      }: _*)
                  }
              )
          )
          .mapGlobals(
            let(__cols = 'global(colsField)) in
              'global
                .insertFields(
                  colsField ->
                    'global('__new_col_idx).map('kv ~>
                      'kv('key).insertFields(
                        child.typ.colValueStruct.fieldNames.map { f =>
                          val s = Symbol(f)
                          s -> 'kv('value).map('i ~> '__cols('i)(s))
                        }: _*
                      ))
                )
                .dropFields('__new_col_idx)
          )

      case MatrixExplodeRows(child, path) =>
        TableExplode(lower(ctx, child, liftedRelationalLets), path)

      case mr: MatrixRead => mr.lower(ctx)

      case MatrixAggregateColsByKey(child, entryExpr, colExpr) =>
        lower(ctx, child, liftedRelationalLets)
          .mapGlobals(
            let(__cols = 'global(colsField)) in
              'global.insertFields(
                '__key_map ->
                  irRange(0, '__cols.len)
                    .map('__old_col_idx ~>
                      (let(__elem = '__cols('__old_col_idx)) in
                        makeStruct(
                          'key -> '__elem.selectFields(child.typ.colKey: _*),
                          'value -> '__old_col_idx,
                        )))
                    .groupByKey
                    .toArray
              )
          )
          .mapRows(
            let(
              __key_map = 'global('__key_map),
              __cols = 'global(colsField),
              __entries = 'row(entriesField),
              global = 'global.selectFields(child.typ.globalType.fieldNames: _*),
              va = 'row.selectFields(child.typ.rowType.fieldNames: _*),
            ) in 'row.insertFields(
              entriesField ->
                irRange(0, '__key_map.len).map('__new_col_idx ~>
                  '__key_map('__new_col_idx)('value).streamAgg('__agg_idx ~>
                    (aggLet(sa = '__cols('__agg_idx), g = '__entries('__agg_idx)) in
                      aggFilter(!'g.isNA, lower(ctx, entryExpr, liftedRelationalLets)))))
            )
          )
          .mapGlobals(
            let(
              __cols = 'global(colsField),
              __key_map = 'global('__key_map),
              global = 'global.selectFields(child.typ.globalType.fieldNames: _*),
            ) in 'global.insertFields(
              colsField ->
                irRange(0, '__key_map.len).map('__new_col_idx ~>
                  (let(__elem = '__key_map('__new_col_idx)) in
                    concatStructs(
                      '__elem('key),
                      '__elem('value).streamAgg('__agg_idx ~>
                        (aggLet(sa = '__cols('__agg_idx)) in
                          lower(ctx, colExpr, liftedRelationalLets))),
                    )))
            )
          )

      case MatrixLiteral(_, tl) => tl
    }

    if (!mir.typ.isCompatibleWith(lowered.typ))
      throw new RuntimeException(
        s"Lowering changed type:\n  BEFORE: ${Pretty(ctx, mir)}\n    ${mir.typ}\n    ${mir.typ.canonicalTableType}\n  AFTER: ${Pretty(ctx, lowered)}\n    ${lowered.typ}"
      )
    lowered
  }

  private def lower(ctx: ExecuteContext, tir: TableIR, ab: Growable[(Name, IR)]): TableIR = {
    val lowered = tir match {
      case CastMatrixToTable(child, entries, cols) =>
        lower(ctx, child, ab)
          .mapRows('row.selectFields(child.typ.rowType.fieldNames :+ entriesFieldName: _*))
          .mapGlobals('global.selectFields(child.typ.globalType.fieldNames :+ colsFieldName: _*))
          .rename(Map(entriesFieldName -> entries), Map(colsFieldName -> cols))

      case x @ MatrixEntriesTable(child) =>
        val lc = lower(ctx, child, ab)

        if (child.typ.rowKey.nonEmpty && child.typ.colKey.nonEmpty) {
          lc
            .mapGlobals(
              let(__cols = 'global(colsField)) in
                'global.insertFields(
                  '__old_col_idx ->
                    irRange(0, '__cols.len)
                      .map('__col_idx ~>
                        makeStruct(
                          'key -> '__cols('__col_idx).selectFields(child.typ.colKey: _*),
                          'value -> '__col_idx,
                        ))
                      .sort(ascending = true, onKey = true)
                      .map('__elem ~> '__elem('value))
                )
            )
            .aggregateByKey(makeStruct(
              '__values ->
                applyAggOp(
                  Collect(),
                  seqOpArgs = FastSeq('row.selectFields(lc.typ.valueType.fieldNames: _*)),
                )
            ))
            .mapRows(
              let(__cols = 'global(colsField)) in
                'row.dropFields('__values).insertFields(
                  '__explode ->
                    'global('__old_col_idx).flatMap('__old_col_idx ~>
                      (let(__col = '__cols('__old_col_idx)) in
                        'row('__values)
                          .filter('__v ~> !'__v(entriesField)('__old_col_idx).isNA)
                          .map('__v ~>
                            (let(__entry = '__v(entriesField)('__old_col_idx)) in
                              makeStruct(
                                child.typ.rowValueStruct.fieldNames.map(Symbol(_)).map(f =>
                                  f -> '__v(f)
                                ) ++
                                  child.typ.colType.fieldNames.map(Symbol(_)).map(f =>
                                    f -> '__col(f)
                                  ) ++
                                  child.typ.entryType.fieldNames.map(Symbol(_)).map(f =>
                                    f -> '__entry(f)
                                  ): _*
                              )))))
                )
            )
            .explode('__explode)
            .mapRows(
              let(__exploded = 'row('__explode)) in
                makeStruct(x.typ.rowType.fieldNames.map { f =>
                  val fd = Symbol(f)
                  (fd, if (child.typ.rowKey.contains(f)) 'row(fd) else '__exploded(fd))
                }: _*)
            )
            .mapGlobals('global.dropFields(colsField, '__old_col_idx))
            .keyBy(child.typ.rowKey ++ child.typ.colKey, isSorted = true)
        } else {
          val result =
            lc
              .mapRows(
                let(__entries = 'row(entriesField)) in
                  'row.insertFields(
                    '__col_idx ->
                      irRange(0, 'global(colsField).len)
                        .filter('__idx ~> !'__entries('__idx).isNA)
                  )
              )
              .explode('__col_idx)
              .mapRows {
                val newFields =
                  child.typ.colType.fieldNames.map(Symbol(_)).map(f => f -> '__col_struct(f)) ++
                    child.typ.entryType.fieldNames.map(Symbol(_)).map(f => f -> '__entry_struct(f))

                let(
                  __col_struct = 'global(colsField)('row('__col_idx)),
                  __entry_struct = 'row(entriesField)('row('__col_idx)),
                ) in 'row
                  .dropFields(entriesField, '__col_idx)
                  .insertFieldsList(newFields, ordering = Some(x.typ.rowType.fieldNames))
              }
              .mapGlobals('global.dropFields(colsField))

          if (child.typ.colKey.isEmpty) result
          else {
            assert(child.typ.rowKey.isEmpty)
            result.keyBy(child.typ.colKey)
          }
        }

      case MatrixToTableApply(child, function) =>
        val loweredChild = lower(ctx, child, ab)
        TableToTableApply(
          loweredChild,
          function.lower()
            .getOrElse(WrappedMatrixToTableFunction(
              function,
              colsFieldName,
              entriesFieldName,
              child.typ.colKey,
            )),
        )

      case MatrixRowsTable(child) =>
        lower(ctx, child, ab)
          .mapGlobals('global.dropFields(colsField))
          .mapRows('row.dropFields(entriesField))

      case MatrixColsTable(child) =>
        val colKey = child.typ.colKey

        val sortedCols =
          if (colKey.isEmpty) '__cols_and_global(colsField)
          else '__cols_and_global(colsField)
            .map('__cols_element ~>
              makeStruct(
                // key struct
                '_1 -> '__cols_element.selectFields(colKey: _*),
                '_2 -> '__cols_element,
              ))
            .sort(true, onKey = true)
            .map('elt ~> 'elt('_2))

        (let(__cols_and_global = lower(ctx, child, ab).getGlobals) in
          makeStruct('rows -> sortedCols, 'global -> '__cols_and_global.dropFields(colsField)))
          .parallelize(None)
          .keyBy(child.typ.colKey)

      case table => lowerChildren(ctx, table, ab).asInstanceOf[TableIR]
    }

    assertTypeUnchanged(tir, lowered)
    lowered
  }

  private def lower(ctx: ExecuteContext, bmir: BlockMatrixIR, ab: Growable[(Name, IR)])
    : BlockMatrixIR = {
    val lowered = lowerChildren(ctx, bmir, ab).asInstanceOf[BlockMatrixIR]
    assertTypeUnchanged(bmir, lowered)
    lowered
  }

  private def lower(ctx: ExecuteContext, ir: IR, ab: Growable[(Name, IR)]): IR = {
    val lowered = ir match {
      case MatrixToValueApply(child, function) =>
        TableToValueApply(
          lower(ctx, child, ab),
          function.lower().getOrElse(
            WrappedMatrixToValueFunction(
              function,
              colsFieldName,
              entriesFieldName,
              child.typ.colKey,
            )
          ),
        )
      case MatrixWrite(child, writer) =>
        TableWrite(
          lower(ctx, child, ab),
          WrappedMatrixWriter(writer, colsFieldName, entriesFieldName, child.typ.colKey),
        )
      case MatrixMultiWrite(children, writer) =>
        TableMultiWrite(
          children.map(lower(ctx, _, ab)),
          WrappedMatrixNativeMultiWriter(writer, children.head.typ.colKey),
        )
      case MatrixCount(child) =>
        lower(ctx, child, ab)
          .aggregate(makeTuple(applyAggOp(Count(), FastSeq(), FastSeq()), 'global(colsField).len))
      case MatrixAggregate(child, query) =>
        val lc = lower(ctx, child, ab)
        TableAggregate(
          lc,
          Let(
            FastSeq(MatrixIR.globalName -> globals(lc)),
            M.agg {
              for {
                cols <- Name("__cols") -> colVals(lc)
                entries <- Name("__entries") -> entries(lc)
                _ <- MatrixIR.globalName -> globals(lc)
                _ <- MatrixIR.rowName -> rowVal(lc)
              } yield aggExplodeIR(
                filterIR(
                  zip2(
                    ToStream(cols),
                    ToStream(entries),
                    ArrayZipBehavior.AssertSameLength,
                  ) {
                    (c, e) => maybeIR(e)(e => maketuple(c, e))
                  }
                )(r => ApplyUnaryPrimOp(Bang, IsNA(r)))
              ) { explodedTuple =>
                M.agg {
                  (MatrixIR.colName -> GetTupleElement(explodedTuple, 0)) >>
                    (MatrixIR.entryName -> GetTupleElement(explodedTuple, 1)) >>
                    query
                }
              }
            },
          ),
        )
      case _ => lowerChildren(ctx, ir, ab).asInstanceOf[IR]
    }
    assertTypeUnchanged(ir, lowered)
    lowered
  }

  private def assertTypeUnchanged(original: BaseIR, lowered: BaseIR): Unit =
    if (lowered.typ != original.typ)
      fatal(
        s"lowering changed type:\n  before: ${original.typ}\n after: ${lowered.typ}\n  ${original.getClass.getName} => ${lowered.getClass.getName}"
      )
}

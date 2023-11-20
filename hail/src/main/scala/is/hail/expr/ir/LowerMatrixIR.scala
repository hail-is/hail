package is.hail.expr.ir

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.functions.{WrappedMatrixToTableFunction, WrappedMatrixToValueFunction}
import is.hail.types._
import is.hail.types.virtual._
import is.hail.utils._

object LowerMatrixIR {
  val entriesFieldName: String = MatrixType.entriesIdentifier
  val colsFieldName: String = "__cols"
  val colsField: Symbol = Symbol(colsFieldName)
  val entriesField: Symbol = Symbol(entriesFieldName)

  def apply(ctx: ExecuteContext, ir: IR): IR = {
    val ab = new BoxedArrayBuilder[(String, IR)]
    val l1 = lower(ctx, ir, ab)
    ab.result().foldRight[IR](l1) { case ((ident, value), body) => RelationalLet(ident, value, body) }
  }

  def apply(ctx: ExecuteContext, tir: TableIR): TableIR = {
    val ab = new BoxedArrayBuilder[(String, IR)]
    val l1 = lower(ctx, tir, ab)
    ab.result().foldRight[TableIR](l1) { case ((ident, value), body) => RelationalLetTable(ident, value, body) }
  }

  def apply(ctx: ExecuteContext, mir: MatrixIR): TableIR = {
    val ab = new BoxedArrayBuilder[(String, IR)]

    val l1 = lower(ctx, mir, ab)
    ab.result().foldRight[TableIR](l1) { case ((ident, value), body) => RelationalLetTable(ident, value, body) }
  }

  def apply(ctx: ExecuteContext, bmir: BlockMatrixIR): BlockMatrixIR = {
    val ab = new BoxedArrayBuilder[(String, IR)]

    val l1 = lower(ctx, bmir, ab)
    ab.result().foldRight[BlockMatrixIR](l1) { case ((ident, value), body) => RelationalLetBlockMatrix(ident, value, body) }
  }


  private[this] def lowerChildren(
    ctx: ExecuteContext,
    ir: BaseIR,
    ab: BoxedArrayBuilder[(String, IR)]
  ): BaseIR = {
    ir.mapChildren {
      case tir: TableIR => lower(ctx, tir, ab)
      case mir: MatrixIR => throw new RuntimeException(s"expect specialized lowering rule for " +
        s"${ ir.getClass.getName }\n  Found MatrixIR child $mir")
      case bmir: BlockMatrixIR => lower(ctx, bmir, ab)
      case vir: IR => lower(ctx, vir, ab)
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

  import is.hail.expr.ir.DeprecatedIRBuilder._

  def matrixSubstEnv(child: MatrixIR): BindingEnv[IRProxy] = {
    val e = Env[IRProxy]("global" -> 'global.selectFields(child.typ.globalType.fieldNames: _*),
      "va" -> 'row.selectFields(child.typ.rowType.fieldNames: _*))
    BindingEnv(e, agg = Some(e), scan = Some(e))
  }

  def matrixGlobalSubstEnv(child: MatrixIR): BindingEnv[IRProxy] = {
    val e = Env[IRProxy]("global" -> 'global.selectFields(child.typ.globalType.fieldNames: _*))
    BindingEnv(e, agg = Some(e), scan = Some(e))
  }

  def matrixSubstEnvIR(child: MatrixIR, lowered: TableIR): BindingEnv[IR] = {
    val e = Env[IR]("global" -> SelectFields(Ref("global", lowered.typ.globalType), child.typ.globalType.fieldNames),
      "va" -> SelectFields(Ref("row", lowered.typ.rowType), child.typ.rowType.fieldNames))
    BindingEnv(e, agg = Some(e), scan = Some(e))
  }


  private[this] def lower(
    ctx: ExecuteContext,
    mir: MatrixIR,
    ab: BoxedArrayBuilder[(String, IR)]
  ): TableIR = {
    val lowered = mir match {
      case RelationalLetMatrixTable(name, value, body) =>
        RelationalLetTable(name, lower(ctx, value, ab), lower(ctx, body, ab))

      case CastTableToMatrix(child, entries, cols, colKey) =>
        val lc = lower(ctx, child, ab)
        val row = Ref("row", lc.typ.rowType)
        val glob = Ref("global", lc.typ.globalType)
        TableMapRows(
          lc,
          bindIR(GetField(row, entries)) { entries =>
            If(IsNA(entries),
              Die("missing entry array unsupported in 'to_matrix_table_row_major'", row.typ),
              bindIRs(ArrayLen(entries), ArrayLen(GetField(glob, cols))) { case Seq(entriesLen, colsLen) =>
                If(entriesLen cne colsLen,
                  Die(
                    strConcat(
                      Str("length mismatch between entry array and column array in 'to_matrix_table_row_major': "),
                      invoke("str", TString, entriesLen),
                      Str(" entries, "),
                      invoke("str", TString, colsLen),
                      Str(" cols, at "),
                      invoke("str", TString, SelectFields(row, child.typ.key))
                    ), row.typ, -1),
                  row
                )
              }
            )
          }
        ).rename(Map(entries -> entriesFieldName), Map(cols -> colsFieldName))

      case MatrixToMatrixApply(child, function) =>
        val loweredChild = lower(ctx, child, ab)
        TableToTableApply(loweredChild, function.lower())

      case MatrixRename(child, globalMap, colMap, rowMap, entryMap) =>
        var t = lower(ctx, child, ab).rename(rowMap, globalMap)

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
        lower(ctx, child, ab).keyBy(keys, isSorted)

      case MatrixFilterRows(child, pred) =>
        lower(ctx, child, ab)
          .filter(subst(lower(ctx, pred, ab), matrixSubstEnv(child)))

      case MatrixFilterCols(child, pred) =>
        lower(ctx, child, ab)
          .mapGlobals('global.insertFields('newColIdx ->
            irRange(0, 'global (colsField).len)
              .filter('i ~>
                (let(sa = 'global (colsField)('i))
                  in subst(lower(ctx, pred, ab), matrixGlobalSubstEnv(child))))))
          .mapRows('row.insertFields(entriesField -> 'global ('newColIdx).map('i ~> 'row (entriesField)('i))))
          .mapGlobals('global
            .insertFields(colsField ->
              'global ('newColIdx).map('i ~> 'global (colsField)('i)))
            .dropFields('newColIdx))

      case MatrixAnnotateRowsTable(child, table, root, product) =>
        val kt = table.typ.keyType
        if (kt.size == 1 && kt.types(0) == TInterval(child.typ.rowKeyStruct.types(0)))
          TableIntervalJoin(lower(ctx, child, ab), lower(ctx, table, ab), root, product)
        else
          TableLeftJoinRightDistinct(lower(ctx, child, ab), lower(ctx, table, ab), root)

      case MatrixChooseCols(child, oldIndices) =>
        lower(ctx, child, ab)
          .mapGlobals('global.insertFields('newColIdx -> oldIndices.map(I32)))
          .mapRows('row.insertFields(entriesField -> 'global ('newColIdx).map('i ~> 'row (entriesField)('i))))
          .mapGlobals('global
            .insertFields(colsField -> 'global ('newColIdx).map('i ~> 'global (colsField)('i)))
            .dropFields('newColIdx))

      case MatrixAnnotateColsTable(child, table, root) =>
        val col = Symbol(genUID())
        val colKey = makeStruct(table.typ.key.zip(child.typ.colKey).map { case (tk, mck) => Symbol(tk) -> col(Symbol(mck)) }: _*)
        lower(ctx, child, ab)
          .mapGlobals(let(__dictfield = lower(ctx, table, ab)
            .keyBy(FastSeq())
            .collect()
            .apply('rows)
            .arrayStructToDict(table.typ.key)) {
            'global.insertFields(colsField ->
              'global (colsField).map(col ~> col.insertFields(Symbol(root) -> '__dictfield.invoke("get", table.typ.valueType, colKey))))
          })

      case MatrixMapGlobals(child, newGlobals) =>
        lower(ctx, child, ab)
          .mapGlobals(
            subst(lower(ctx, newGlobals, ab), BindingEnv(Env[IRProxy](
              "global" -> 'global.selectFields(child.typ.globalType.fieldNames: _*))))
              .insertFields(colsField -> 'global (colsField)))

      case MatrixMapRows(child, newRow) =>
        def liftScans(ir: IR): IRProxy = {
          def lift(ir: IR, builder: BoxedArrayBuilder[(String, IR)]): IR = ir match {
            case a: ApplyScanOp =>
              val s = genUID()
              builder += ((s, a))
              Ref(s, a.typ)

            case a@AggFold(zero, seqOp, combOp, accumName, otherAccumName, true) =>
              val s = genUID()
              builder += ((s, a))
              Ref(s, a.typ)

            case AggFilter(filt, body, true) =>
              val ab = new BoxedArrayBuilder[(String, IR)]
              val liftedBody = lift(body, ab)
              val uid = genUID()
              val aggs = ab.result()
              val structResult = MakeStruct(aggs)
              val aggFilterIR = AggFilter(filt, structResult, true)
              builder += ((uid, aggFilterIR))
              aggs.foldLeft[IR](liftedBody) { case (acc, (name, _)) => Let(name, GetField(Ref(uid, structResult.typ), name), acc) }

            case AggExplode(a, name, body, true) =>
              val ab = new BoxedArrayBuilder[(String, IR)]
              val liftedBody = lift(body, ab)
              val uid = genUID()
              val aggs = ab.result()
              val structResult = MakeStruct(aggs)
              val aggExplodeIR = AggExplode(a, name, structResult, true)
              builder += ((uid, aggExplodeIR))
              aggs.foldLeft[IR](liftedBody) { case (acc, (name, _)) => Let(name, GetField(Ref(uid, structResult.typ), name), acc) }

            case AggGroupBy(a, body, true) =>
              val ab = new BoxedArrayBuilder[(String, IR)]
              val liftedBody = lift(body, ab)
              val uid = genUID()
              val aggs = ab.result()
              val structResult = MakeStruct(aggs)
              val aggIR = AggGroupBy(a, structResult, true)
              builder += ((uid, aggIR))
              val eltUID = genUID()
              val valueUID = genUID()
              val elementType = aggIR.typ.asInstanceOf[TDict].elementType
              val valueType = elementType.asInstanceOf[TBaseStruct].types(1)
              ToDict(StreamMap(ToStream(Ref(uid, aggIR.typ)), eltUID, Let(valueUID, GetField(Ref(eltUID, elementType), "value"),
                MakeTuple.ordered(FastSeq(GetField(Ref(eltUID, elementType), "key"),
                  aggs.foldLeft[IR](liftedBody) { case (acc, (name, _)) => Let(name, GetField(Ref(valueUID, valueType), name), acc) })))))

            case AggArrayPerElement(a, elementName, indexName, body, knownLength, true) =>
              val ab = new BoxedArrayBuilder[(String, IR)]
              val liftedBody = lift(body, ab)
              val uid = genUID()
              val aggs = ab.result()
              val structResult = MakeStruct(aggs)
              val aggIR = AggArrayPerElement(a, elementName, indexName, structResult, knownLength, true)
              builder += ((uid, aggIR))
              val eltUID = genUID()
              val t = aggIR.typ.asInstanceOf[TArray]
              ToArray(StreamMap(ToStream(Ref(uid, t)), eltUID, aggs.foldLeft[IR](liftedBody) { case (acc, (name, _)) => Let(name, GetField(Ref(eltUID, structResult.typ), name), acc) }))

            case AggLet(name, value, body, true) =>
              val ab = new BoxedArrayBuilder[(String, IR)]
              val liftedBody = lift(body, ab)
              val uid = genUID()
              val aggs = ab.result()
              val structResult = MakeStruct(aggs)
              val aggIR = AggLet(name, value, structResult, true)
              builder += ((uid, aggIR))
              aggs.foldLeft[IR](liftedBody) { case (acc, (name, _)) => Let(name, GetField(Ref(uid, structResult.typ), name), acc) }

            case _ =>
              MapIR(lift(_, builder))(ir)
          }

          val ab = new BoxedArrayBuilder[(String, IR)]
          val b0 = lift(ir, ab)

          val scans = ab.result()
          val scanStruct = MakeStruct(scans)

          val scanResultRef = Ref(genUID(), scanStruct.typ)

          val b1 = if (ContainsAgg(b0)) {
            irRange(0, 'row(entriesField).len)
              .filter('i ~> !'row(entriesField)('i).isNA)
              .streamAgg('i ~>
                (aggLet(sa = 'global(colsField)('i),
                  g = 'row(entriesField)('i))
                  in b0))
          } else
            irToProxy(b0)

          let.applyDynamicNamed("apply")((scanResultRef.name, scanStruct))(
            scans.foldLeft[IRProxy](b1) { case (acc, (name, _)) => let.applyDynamicNamed("apply")((name, GetField(scanResultRef, name)))(acc) })
        }


        val lc = lower(ctx, child, ab)
        lc.mapRows(let(n_cols = 'global(colsField).len) {
          liftScans(Subst(lower(ctx, newRow, ab), matrixSubstEnvIR(child, lc)))
            .insertFields(entriesField -> 'row(entriesField))
        })

      case MatrixMapCols(child, newCol, _) =>
        val loweredChild = lower(ctx, child, ab)

        def lift(ir: IR, scanBindings: BoxedArrayBuilder[(String, IR)], aggBindings: BoxedArrayBuilder[(String, IR)]): IR = ir match {
          case a: ApplyScanOp =>
            val s = genUID()
            scanBindings += ((s, a))
            Ref(s, a.typ)

          case a: ApplyAggOp =>
            val s = genUID()
            aggBindings += ((s, a))
            Ref(s, a.typ)

          case a@AggFold(zero, seqOp, combOp, accumName, otherAccumName, isScan) =>
            val s = genUID()
            if (isScan) {
              scanBindings += ((s, a))
            } else {
              aggBindings += ((s, a))
            }
            Ref(s, a.typ)

          case AggFilter(filt, body, isScan) =>
            val ab = new BoxedArrayBuilder[(String, IR)]
            val (liftedBody, builder) = if (isScan)
              (lift(body, ab, aggBindings), scanBindings)
            else
              (lift(body, scanBindings, ab), aggBindings)
            val uid = genUID()
            val aggs = ab.result()
            val structResult = MakeStruct(aggs)
            val aggFilterIR = AggFilter(filt, structResult, isScan)
            builder += ((uid, aggFilterIR))
            aggs.foldLeft[IR](liftedBody) { case (acc, (name, _)) => Let(name, GetField(Ref(uid, structResult.typ), name), acc) }

          case AggExplode(a, name, body, isScan) =>
            val ab = new BoxedArrayBuilder[(String, IR)]
            val (liftedBody, builder) = if (isScan)
              (lift(body, ab, aggBindings), scanBindings)
            else
              (lift(body, scanBindings, ab), aggBindings)
            val uid = genUID()
            val aggs = ab.result()
            val structResult = MakeStruct(aggs)
            val aggExplodeIR = AggExplode(a, name, structResult, isScan)
            builder += ((uid, aggExplodeIR))
            aggs.foldLeft[IR](liftedBody) { case (acc, (name, _)) => Let(name, GetField(Ref(uid, structResult.typ), name), acc) }

          case AggGroupBy(a, body, isScan) =>
            val ab = new BoxedArrayBuilder[(String, IR)]
            val (liftedBody, builder) = if (isScan)
              (lift(body, ab, aggBindings), scanBindings)
            else
              (lift(body, scanBindings, ab), aggBindings)
            val uid = genUID()
            val aggs = ab.result()
            val structResult = MakeStruct(aggs)
            val aggIR = AggGroupBy(a, structResult, isScan)
            builder += ((uid, aggIR))
            val eltUID = genUID()
            val valueUID = genUID()
            val elementType = aggIR.typ.asInstanceOf[TDict].elementType
            val valueType = elementType.asInstanceOf[TBaseStruct].types(1)
            ToDict(StreamMap(ToStream(Ref(uid, aggIR.typ)), eltUID, Let(valueUID, GetField(Ref(eltUID, elementType), "value"),
              MakeTuple.ordered(FastSeq(GetField(Ref(eltUID, elementType), "key"),
                aggs.foldLeft[IR](liftedBody) { case (acc, (name, _)) => Let(name, GetField(Ref(valueUID, valueType), name), acc) } )))))

          case AggArrayPerElement(a, elementName, indexName, body, knownLength, isScan) =>
            val ab = new BoxedArrayBuilder[(String, IR)]
            val (liftedBody, builder) = if (isScan)
              (lift(body, ab, aggBindings), scanBindings)
            else
              (lift(body, scanBindings, ab), aggBindings)
            val uid = genUID()
            val aggs = ab.result()
            val structResult = MakeStruct(aggs)
            val aggIR = AggArrayPerElement(a, elementName, indexName, structResult, knownLength, isScan)
            builder += ((uid, aggIR))
            val eltUID = genUID()
            val t = aggIR.typ.asInstanceOf[TArray]
            ToArray(StreamMap(ToStream(Ref(uid, t)), eltUID, aggs.foldLeft[IR](liftedBody) { case (acc, (name, _)) => Let(name, GetField(Ref(eltUID, structResult.typ), name), acc) }))

          case AggLet(name, value, body, isScan) =>
            val ab = new BoxedArrayBuilder[(String, IR)]
            val (liftedBody, builder) = if (isScan)
              (lift(body, ab, aggBindings), scanBindings)
            else
              (lift(body, scanBindings, ab), aggBindings)
            val uid = genUID()
            val aggs = ab.result()
            val structResult = MakeStruct(aggs)
            val aggIR = AggLet(name, value, structResult, isScan)
            builder += ((uid, aggIR))
            aggs.foldLeft[IR](liftedBody) { case (acc, (name, _)) => Let(name, GetField(Ref(uid, structResult.typ), name), acc) }

          case x: StreamAgg => x
          case x: StreamAggScan => x

          case _ =>
            MapIR(lift(_, scanBindings, aggBindings))(ir)
        }

        val scanBuilder = new BoxedArrayBuilder[(String, IR)]
        val aggBuilder = new BoxedArrayBuilder[(String, IR)]

        val b0 = lift(Subst(lower(ctx, newCol, ab), matrixSubstEnvIR(child, loweredChild)), scanBuilder, aggBuilder)
        val aggs = aggBuilder.result()
        val scans = scanBuilder.result()

        val idx = Ref(genUID(), TInt32)
        val idxSym = Symbol(idx.name)

        val noOp: (IRProxy => IRProxy, IRProxy => IRProxy) = (identity[IRProxy], identity[IRProxy])

        val (aggOutsideTransformer: (IRProxy => IRProxy), aggInsideTransformer: (IRProxy => IRProxy)) = if (aggs.isEmpty)
          noOp
        else {
          val aggStruct = MakeStruct(aggs)

          val aggResult = loweredChild.aggregate(
            aggLet(va = 'row.selectFields(child.typ.rowType.fieldNames: _*)) {
              makeStruct(
                ('count, applyAggOp(Count(), FastSeq(), FastSeq())),
                ('array_aggs, irRange(0, 'global(colsField).len)
                  .aggElements('__element_idx, '__result_idx, Some('global(colsField).len))(
                    let(sa = 'global(colsField)('__result_idx)) {
                      aggLet(sa = 'global(colsField)('__element_idx),
                        g = 'row(entriesField)('__element_idx)) {
                        aggFilter(!'g.isNA, aggStruct)
                      }
                    })))
            })

          val ident = genUID()
          ab += ((ident, aggResult))

          val aggResultRef = Ref(genUID(), aggResult.typ)
          val aggResultElementRef = Ref(genUID(), aggResult.typ.asInstanceOf[TStruct]
            .fieldType("array_aggs")
            .asInstanceOf[TArray].elementType)

          val bindResult: IRProxy => IRProxy = let.applyDynamicNamed("apply")((aggResultRef.name, irToProxy(RelationalRef(ident, aggResult.typ)))).apply(_)
          val bodyResult: IRProxy => IRProxy = (x: IRProxy) =>
            let.applyDynamicNamed("apply")((aggResultRef.name, irToProxy(RelationalRef(ident, aggResult.typ))))
              .apply(let(n_rows = Symbol(aggResultRef.name)('count), array_aggs = Symbol(aggResultRef.name)('array_aggs)) {
                let.applyDynamicNamed("apply")((aggResultElementRef.name, 'array_aggs(idx))) {
                  aggs.foldLeft[IRProxy](x) { case (acc, (name, _)) => let.applyDynamicNamed("apply")((name, GetField(aggResultElementRef, name)))(acc) }
                }
              })
          (bindResult, bodyResult)
        }

        val (scanOutsideTransformer: (IRProxy => IRProxy), scanInsideTransformer: (IRProxy => IRProxy)) = if (scans.isEmpty)
          noOp
        else {
          val scanStruct = MakeStruct(scans)

          val scanResultArray = ToArray(StreamAggScan(
            ToStream(GetField(Ref("global", loweredChild.typ.globalType), colsFieldName)),
            "sa",
            scanStruct))

          val scanResultRef = Ref(genUID(), scanResultArray.typ)
          val scanResultElementRef = Ref(genUID(), scanResultArray.typ.asInstanceOf[TArray].elementType)

          val bindResult: IRProxy => IRProxy = let.applyDynamicNamed("apply")((scanResultRef.name, scanResultArray)).apply(_)
          val bodyResult: IRProxy => IRProxy = (x: IRProxy) =>
            let.applyDynamicNamed("apply")((scanResultElementRef.name, ArrayRef(scanResultRef, idx)))(
              scans.foldLeft[IRProxy](x) { case (acc, (name, _)) =>
                let.applyDynamicNamed("apply")((name, GetField(scanResultElementRef, name)))(acc)
              })
          (bindResult, bodyResult)
        }

        loweredChild.mapGlobals('global.insertFields(colsField ->
          aggOutsideTransformer(scanOutsideTransformer(irRange(0, 'global(colsField).len).map(idxSym ~> let(__cols_array = 'global(colsField), sa = '__cols_array(idxSym)) {
            aggInsideTransformer(scanInsideTransformer(b0))
          })))
        ))

      case MatrixFilterEntries(child, pred) =>
        val lc = lower(ctx, child, ab)
          lc.mapRows('row.insertFields(entriesField ->
          irRange(0, 'global (colsField).len).map {
            'i ~>
              let(g = 'row (entriesField)('i)) {
                irIf(let(sa = 'global (colsField)('i))
                  in !subst(lower(ctx, pred, ab), matrixSubstEnv(child))) {
                  NA(child.typ.entryType)
                } {
                  'g
                }
              }
          }))

      case MatrixUnionCols(left, right, joinType) =>
        val rightEntries = genUID()
        val rightCols = genUID()
        val ll = lower(ctx, left, ab).distinct()
        def handleMissingEntriesArray(entries: Symbol, cols: Symbol): IRProxy =
          if (joinType == "inner")
            'row(entries)
          else
            irIf('row(entries).isNA) {
              irRange(0, 'global(cols).len)
                .map('a ~> irToProxy(MakeStruct(right.typ.entryType.fieldNames.map(f => (f, NA(right.typ.entryType.fieldType(f)))))))
            } {
              'row(entries)
            }
        val rr = lower(ctx, right, ab).distinct()
        TableJoin(
          ll,
          rr.mapRows('row.castRename(rr.typ.rowType.rename(Map(entriesFieldName -> rightEntries))))
            .mapGlobals('global
              .insertFields(Symbol(rightCols) -> 'global(colsField))
              .selectFields(rightCols)),
          joinType)
          .mapRows('row
            .insertFields(entriesField ->
              makeArray(
                handleMissingEntriesArray(entriesField, colsField),
                handleMissingEntriesArray(Symbol(rightEntries), Symbol(rightCols)))
                .flatMap('a ~> 'a))
            .dropFields(Symbol(rightEntries)))
          .mapGlobals('global
            .insertFields(colsField ->
              makeArray('global(colsField), 'global(Symbol(rightCols))).flatMap('a ~> 'a))
            .dropFields(Symbol(rightCols)))

      case MatrixMapEntries(child, newEntries) =>
        val loweredChild = lower(ctx, child, ab)
        val rt = loweredChild.typ.rowType
        val gt = loweredChild.typ.globalType
        TableMapRows(
          loweredChild,
          InsertFields(
            Ref("row", rt),
            FastSeq((entriesFieldName, ToArray(StreamZip(
              FastSeq(
                ToStream(GetField(Ref("row", rt), entriesFieldName)),
                ToStream(GetField(Ref("global", gt), colsFieldName))),
              FastSeq("g", "sa"),
              Subst(lower(ctx, newEntries, ab), BindingEnv(Env(
                "global" -> SelectFields(Ref("global", gt), child.typ.globalType.fieldNames),
                "va" -> SelectFields(Ref("row", rt), child.typ.rowType.fieldNames)))),
              ArrayZipBehavior.AssumeSameLength
            )))))
        )

      case MatrixRepartition(child, n, shuffle) => TableRepartition(lower(ctx, child, ab), n, shuffle)

      case MatrixFilterIntervals(child, intervals, keep) => TableFilterIntervals(lower(ctx, child, ab), intervals, keep)

      case MatrixUnionRows(children) =>
        // FIXME: this should check that all children have the same column keys.
        val first = lower(ctx, children.head, ab)
        TableUnion(FastSeq(first) ++
          children.tail.map(lower(ctx, _, ab)
            .mapRows('row.selectFields(first.typ.rowType.fieldNames: _*))))

      case MatrixDistinctByRow(child) => TableDistinct(lower(ctx, child, ab))

      case MatrixRowsHead(child, n) => TableHead(lower(ctx, child, ab), n)
      case MatrixRowsTail(child, n) => TableTail(lower(ctx, child, ab), n)

      case MatrixColsHead(child, n) => lower(ctx, child, ab)
        .mapGlobals('global.insertFields(colsField -> 'global (colsField).arraySlice(0, Some(n), 1)))
        .mapRows('row.insertFields(entriesField -> 'row (entriesField).arraySlice(0, Some(n), 1)))

      case MatrixColsTail(child, n) => lower(ctx, child, ab)
        .mapGlobals('global.insertFields(colsField -> 'global (colsField).arraySlice(-n, None, 1)))
        .mapRows('row.insertFields(entriesField -> 'row (entriesField).arraySlice(-n, None, 1)))

      case MatrixExplodeCols(child, path) =>
        val loweredChild = lower(ctx, child, ab)
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
        val eeSub = subst(lower(ctx, entryExpr, ab), substEnv)
        val reSub = subst(lower(ctx, rowExpr, ab), substEnv)
        lower(ctx, child, ab)
          .aggregateByKey(
            reSub.insertFields(entriesField -> irRange(0, 'global (colsField).len)
              .aggElements('__element_idx, '__result_idx, Some('global (colsField).len))(
                let(sa = 'global (colsField)('__result_idx)) {
                  aggLet(sa = 'global (colsField)('__element_idx),
                    g = 'row (entriesField)('__element_idx)) {
                    aggFilter(!'g.isNA, eeSub)
                  }
                })))

      case MatrixCollectColsByKey(child) =>
        lower(ctx, child, ab)
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

      case MatrixExplodeRows(child, path) => TableExplode(lower(ctx, child, ab), path)

      case mr: MatrixRead => mr.lower()

      case MatrixAggregateColsByKey(child, entryExpr, colExpr) =>
        val colKey = child.typ.colKey

        val originalColIdx = Symbol(genUID())
        val newColIdx1 = Symbol(genUID())
        val newColIdx2 = Symbol(genUID())
        val colsAggIdx = Symbol(genUID())
        val keyMap = Symbol(genUID())
        val aggElementIdx = Symbol(genUID())

        val e1 = Env[IRProxy]("global" -> 'global.selectFields(child.typ.globalType.fieldNames: _*),
          "va" -> 'row.selectFields(child.typ.rowType.fieldNames: _*))
        val e2 = Env[IRProxy]("global" -> 'global.selectFields(child.typ.globalType.fieldNames: _*))
        val ceSub = subst(lower(ctx, colExpr, ab), BindingEnv(e2, agg = Some(e1)))
        val eeSub = subst(lower(ctx, entryExpr, ab), BindingEnv(e1, agg = Some(e1)))

        lower(ctx, child, ab)
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
                  .streamAgg(aggElementIdx ~>
                    aggLet(g = '__entries (aggElementIdx), sa = 'global (colsField)(aggElementIdx)) {
                      aggFilter(!'g.isNA, eeSub)
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
                        .streamAgg(colsAggIdx ~> aggLet(sa = 'global (colsField)(colsAggIdx)) {
                          ceSub
                        })
                    ))
              }
            ).dropFields(keyMap))

      case MatrixLiteral(_, tl) => tl
    }

    if (!mir.typ.isCompatibleWith(lowered.typ))
      throw new RuntimeException(s"Lowering changed type:\n  BEFORE: ${ Pretty(ctx, mir) }\n    ${ mir.typ }\n    ${ mir.typ.canonicalTableType}\n  AFTER: ${ Pretty(ctx, lowered) }\n    ${ lowered.typ }")
    lowered
  }


  private[this] def lower(
    ctx: ExecuteContext,
    tir: TableIR,
    ab: BoxedArrayBuilder[(String, IR)]
  ): TableIR = {
    val lowered = tir match {
      case CastMatrixToTable(child, entries, cols) =>
        lower(ctx, child, ab)
          .mapRows('row.selectFields(child.typ.rowType.fieldNames ++ Array(entriesFieldName): _*))
          .mapGlobals('global.selectFields(child.typ.globalType.fieldNames ++ Array(colsFieldName):  _*))
          .rename(Map(entriesFieldName -> entries), Map(colsFieldName -> cols))

      case x@MatrixEntriesTable(child) =>
        val lc = lower(ctx, child, ab)

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
            .aggregateByKey(makeStruct(values -> applyAggOp(Collect(), seqOpArgs = FastSeq('row.selectFields(lc.typ.valueType.fieldNames: _*)))))
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
                ordering = Some(x.typ.rowType.fieldNames.toFastSeq))
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
        val loweredChild = lower(ctx, child, ab)
        TableToTableApply(loweredChild,
          function.lower()
            .getOrElse(WrappedMatrixToTableFunction(function, colsFieldName, entriesFieldName, child.typ.colKey)))

      case MatrixRowsTable(child) =>
        lower(ctx, child, ab)
          .mapGlobals('global.dropFields(colsField))
          .mapRows('row.dropFields(entriesField))

      case MatrixColsTable(child) =>
        val colKey = child.typ.colKey
        let(__cols_and_globals = lower(ctx, child, ab).getGlobals) {
          val sortedCols = if (colKey.isEmpty)
            '__cols_and_globals (colsField)
          else
            '__cols_and_globals (colsField).map { '__cols_element ~>
              makeStruct(
                // key struct
                '_1 -> '__cols_element.selectFields(colKey: _*),
                '_2 -> '__cols_element)
            }.sort(true, onKey = true)
              .map {
                'elt ~> 'elt ('_2)
              }
          makeStruct('rows -> sortedCols, 'global -> '__cols_and_globals.dropFields(colsField))
        }.parallelize(None).keyBy(child.typ.colKey)

      case table => lowerChildren(ctx, table, ab).asInstanceOf[TableIR]
    }

    assertTypeUnchanged(tir, lowered)
    lowered
  }

  private[this] def lower(
    ctx: ExecuteContext,
    bmir: BlockMatrixIR,
    ab: BoxedArrayBuilder[(String, IR)]
  ): BlockMatrixIR = {
    val lowered = bmir match {
      case noMatrixChildren => lowerChildren(ctx, noMatrixChildren, ab).asInstanceOf[BlockMatrixIR]
    }
    assertTypeUnchanged(bmir, lowered)
    lowered
  }

  private[this] def lower(
    ctx: ExecuteContext,
    ir: IR,
    ab: BoxedArrayBuilder[(String, IR)]
  ): IR = {
    val lowered = ir match {
      case MatrixToValueApply(child, function) => TableToValueApply(lower(ctx, child, ab), function.lower()
        .getOrElse(WrappedMatrixToValueFunction(function, colsFieldName, entriesFieldName, child.typ.colKey)))
      case MatrixWrite(child, writer) =>
        TableWrite(lower(ctx, child, ab), WrappedMatrixWriter(writer, colsFieldName, entriesFieldName, child.typ.colKey))
      case MatrixMultiWrite(children, writer) =>
        TableMultiWrite(children.map(lower(ctx, _, ab)), WrappedMatrixNativeMultiWriter(writer, children.head.typ.colKey))
      case MatrixCount(child) =>
        lower(ctx, child, ab)
          .aggregate(makeTuple(applyAggOp(Count(), FastSeq(), FastSeq()), 'global(colsField).len))
      case MatrixAggregate(child, query) =>
        val lc = lower(ctx, child, ab)
        val idx = Symbol(genUID())
        TableAggregate(lc,
          aggExplodeIR(
            filterIR(
              zip2(
                ToStream(GetField(Ref("row", lc.typ.rowType), entriesFieldName)),
                ToStream(GetField(Ref("global", lc.typ.globalType), colsFieldName)),
                ArrayZipBehavior.AssertSameLength
              ) { case (e, c) =>
                MakeTuple.ordered(FastSeq(e, c))
              }) { filterTuple =>
              ApplyUnaryPrimOp(Bang, IsNA(GetTupleElement(filterTuple, 0)))
            }) { explodedTuple =>
            AggLet("g", GetTupleElement(explodedTuple, 0),
              AggLet("sa", GetTupleElement(explodedTuple, 1), Subst(query, matrixSubstEnvIR(child, lc)),
                isScan = false),
              isScan = false)
          })
      case _ => lowerChildren(ctx, ir, ab).asInstanceOf[IR]
    }
    assertTypeUnchanged(ir, lowered)
    lowered
  }

  private[this] def assertTypeUnchanged(original: BaseIR, lowered: BaseIR) {
    if (lowered.typ != original.typ)
      fatal(s"lowering changed type:\n  before: ${ original.typ }\n after: ${ lowered.typ }\n  ${ original.getClass.getName } => ${ lowered.getClass.getName }")
  }
}

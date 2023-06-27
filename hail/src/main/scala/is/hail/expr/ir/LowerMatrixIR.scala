package is.hail.expr.ir

import cats.implicits.toTraverseOps
import cats.syntax.all._
import cats.{Applicative, MonadThrow}
import is.hail.expr.ir.functions.{WrappedMatrixToTableFunction, WrappedMatrixToValueFunction}
import is.hail.expr.ir.lowering.MonadLower
import is.hail.types._
import is.hail.types.virtual._
import is.hail.utils._

import scala.language.higherKinds

object LowerMatrixIR {
  val entriesFieldName: String = MatrixType.entriesIdentifier
  val colsFieldName: String = "__cols"
  val colsField: Symbol = Symbol(colsFieldName)
  val entriesField: Symbol = Symbol(entriesFieldName)

  def apply[M[_]: MonadLower, A <: BaseIR](ir: A): M[BaseIR] =
    ir match {
      case ir: IR => apply(ir, lowerIR[M], RelationalLet).widen[BaseIR]
      case ir: TableIR => apply(ir, lowerTableIR[M], RelationalLetTable).widen[BaseIR]
      case ir: MatrixIR => apply(ir, lowerMatrixIR[M], RelationalLetTable).widen[BaseIR]
      case ir: BlockMatrixIR => apply(ir, lowerBlockMatrixIR[M], RelationalLetBlockMatrix).widen[BaseIR]
    }

  def apply[M[_]: MonadLower, A, B](ir: A,
                                    lower: (A, BoxedArrayBuilder[(String, IR)]) => M[B],
                                    let: (String, IR, B) => B
                                   ): M[B] = {
    val ab = new BoxedArrayBuilder[(String, IR)]
    lower(ir, ab).map {
      ab.result().foldRight(_) { case ((ident, value), body) => let(ident, value, body) }
    }
  }

  private[this] def lowerChildren[M[_]: MonadLower](ir: BaseIR, ab: BoxedArrayBuilder[(String, IR)])
  : M[BaseIR] =
    ir.traverseChildren {
      case vir: IR => lowerIR(vir, ab).widen[BaseIR]
      case tir: TableIR => lowerTableIR(tir, ab).widen[BaseIR]
      case bmir: BlockMatrixIR => lowerBlockMatrixIR(bmir, ab).widen[BaseIR]
      case mir: MatrixIR =>
        MonadLower[M].raiseError(
          new RuntimeException(
            s"expect specialized lowering rule for ${ir.getClass.getName}\n  Found MatrixIR child $mir"
          )
        )
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


  private[this] def lowerMatrixIR[M[_]: MonadLower](mir: MatrixIR, ab: BoxedArrayBuilder[(String, IR)])
  : M[TableIR] = {
    val lowered: M[TableIR] = mir match {
      case RelationalLetMatrixTable(name, value, body) =>
        for {
          loweredValue <- lowerIR(value, ab)
          loweredBody <- lowerMatrixIR(body, ab)
        } yield RelationalLetTable(name, loweredValue, loweredBody)

      case CastTableToMatrix(child, entries, cols, colKey) =>
        lowerTableIR(child, ab).map { lc =>
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
        }

      case MatrixToMatrixApply(child, function) =>
        lowerMatrixIR(child, ab).map {
          TableToTableApply(_, function.lower())
        }

      case MatrixRename(child, globalMap, colMap, rowMap, entryMap) =>
        lowerMatrixIR(child, ab).map { t0 =>
          var t = t0.rename(rowMap, globalMap)

          if (colMap.nonEmpty) {
            val newColsType = TArray(child.typ.colType.rename(colMap))
            t = t.mapGlobals('global.castRename(t.typ.globalType.insertFields(FastSeq((colsFieldName, newColsType)))))
          }

          if (entryMap.nonEmpty) {
            val newEntriesType = TArray(child.typ.entryType.rename(entryMap))
            t = t.mapRows('row.castRename(t.typ.rowType.insertFields(FastSeq((entriesFieldName, newEntriesType)))))
          }

          t
        }

      case MatrixKeyRowsBy(child, keys, isSorted) =>
        lowerMatrixIR(child, ab).map(_.keyBy(keys, isSorted))

      case MatrixFilterRows(child, pred) =>
        for {
          lc <- lowerMatrixIR(child, ab)
          lp <- lowerIR(pred, ab)
        } yield lc.filter(subst(lp, matrixSubstEnv(child)))

      case MatrixFilterCols(child, pred) =>
        for {
          lc <- lowerMatrixIR(child, ab)
          lp <- lowerIR(pred, ab)
        } yield lc.mapGlobals('global.insertFields('newColIdx ->
          irRange(0, 'global(colsField).len)
            .filter('i ~>
              (let(sa = 'global(colsField)('i))
                in subst(lp, matrixGlobalSubstEnv(child))))))
          .mapRows('row.insertFields(entriesField -> 'global('newColIdx).map('i ~> 'row(entriesField)('i))))
          .mapGlobals('global
            .insertFields(colsField ->
              'global('newColIdx).map('i ~> 'global(colsField)('i)))
            .dropFields('newColIdx))

      case MatrixAnnotateRowsTable(child, table, root, product) =>
        for {
          loweredChild <- lowerMatrixIR(child, ab)
          loweredTable <- lowerTableIR(table, ab)
          kt = table.typ.keyType
        } yield if (kt.size == 1 && kt.types(0) == TInterval(child.typ.rowKeyStruct.types(0)))
            TableIntervalJoin(loweredChild, loweredTable, root, product)
          else
            TableLeftJoinRightDistinct(loweredChild, loweredTable, root)

      case MatrixChooseCols(child, oldIndices) =>
        lowerMatrixIR(child, ab).map {
          _.mapGlobals('global.insertFields('newColIdx -> oldIndices.map(I32)))
            .mapRows('row.insertFields(entriesField -> 'global('newColIdx).map('i ~> 'row(entriesField)('i))))
            .mapGlobals('global
              .insertFields(colsField -> 'global('newColIdx).map('i ~> 'global(colsField)('i)))
              .dropFields('newColIdx))
        }

      case MatrixAnnotateColsTable(child, table, root) =>
        for {
          lowererChild <- lowerMatrixIR(child, ab)
          loweredTable <- lowerTableIR(table, ab)
        } yield {
          val col = Symbol(genUID())
          val colKey = makeStruct(table.typ.key.zip(child.typ.colKey).map { case (tk, mck) => Symbol(tk) -> col(Symbol(mck)) }: _*)
          lowererChild.mapGlobals(
            let(__dictfield = loweredTable
              .keyBy(FastIndexedSeq())
              .collect()
              .apply('rows)
              .arrayStructToDict(table.typ.key)
            ) {
              'global.insertFields(colsField ->
                'global(colsField).map(col ~>
                  col.insertFields(Symbol(root) ->
                    '__dictfield.invoke("get", table.typ.valueType, colKey)
                  )
                )
              )
            }
          )
        }

      case MatrixMapGlobals(child, newGlobals) =>
        for {
          loweredChild <- lowerMatrixIR(child, ab)
          loweredGlobals <- lowerIR(newGlobals, ab)
        } yield loweredChild.mapGlobals(
          subst(
            loweredGlobals,
            BindingEnv(Env[IRProxy]("global" ->
              'global.selectFields(child.typ.globalType.fieldNames: _*)
            ))
          ).insertFields(colsField -> 'global(colsField))
        )

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

        for {
          lc <- lowerMatrixIR(child, ab)
          loweredRow <- lowerIR(newRow, ab)
        } yield lc.mapRows {
          let(n_cols = 'global(colsField).len) {
            liftScans(Subst(loweredRow, matrixSubstEnvIR(child, lc)))
              .insertFields(entriesField -> 'row(entriesField))
          }
        }

      case MatrixMapCols(child, newCol, _) =>
        for {
          loweredChild <- lowerMatrixIR(child, ab)
          loweredCol <- lowerIR(newCol, ab)
        } yield {
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
                    aggs.foldLeft[IR](liftedBody) { case (acc, (name, _)) => Let(name, GetField(Ref(valueUID, valueType), name), acc) })))))

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

              case _ =>
                MapIR(lift(_, scanBindings, aggBindings))(ir)
            }

            val scanBuilder = new BoxedArrayBuilder[(String, IR)]
            val aggBuilder = new BoxedArrayBuilder[(String, IR)]

            val b0 = lift(Subst(loweredCol, matrixSubstEnvIR(child, loweredChild)), scanBuilder, aggBuilder)
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
                    ('count, applyAggOp(Count(), FastIndexedSeq(), FastIndexedSeq())),
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
          }

      case MatrixFilterEntries(child, pred) =>
        for {
          lc <- lowerMatrixIR(child, ab)
          lp <- lowerIR(pred, ab)
        } yield lc.mapRows('row.insertFields(entriesField ->
          irRange(0, 'global(colsField).len).map {
            'i ~>
              let(g = 'row(entriesField)('i)) {
                irIf(let(sa = 'global(colsField)('i))
                  in !subst(lp, matrixSubstEnv(child))) {
                  NA(child.typ.entryType)
                } {
                  'g
                }
              }
          }))

      case MatrixUnionCols(left, right, joinType) =>
        for {
          loweredLeft <- lowerMatrixIR(left, ab)
          loweredRight <- lowerMatrixIR(right, ab)
        } yield {
          val rightEntries = genUID()
          val rightCols = genUID()
          val ll = loweredLeft.distinct()

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

          val rr = loweredRight.distinct()
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
        }

      case MatrixMapEntries(child, newEntries) =>
        for {
          loweredChild <- lowerMatrixIR(child, ab)
          loweredEntries <- lowerIR(newEntries, ab)
        } yield {
          val rt = loweredChild.typ.rowType
          val gt = loweredChild.typ.globalType
          TableMapRows(
            loweredChild,
            InsertFields(
              Ref("row", rt),
              FastSeq((entriesFieldName, ToArray(StreamZip(
                FastIndexedSeq(
                  ToStream(GetField(Ref("row", rt), entriesFieldName)),
                  ToStream(GetField(Ref("global", gt), colsFieldName))),
                FastIndexedSeq("g", "sa"),
                Subst(loweredEntries, BindingEnv(Env(
                  "global" -> SelectFields(Ref("global", gt), child.typ.globalType.fieldNames),
                  "va" -> SelectFields(Ref("row", rt), child.typ.rowType.fieldNames)))),
                ArrayZipBehavior.AssumeSameLength
              ))))))
        }

      case MatrixRepartition(child, n, shuffle) =>
        lowerMatrixIR(child, ab).map {
          TableRepartition(_, n, shuffle)
        }

      case MatrixFilterIntervals(child, intervals, keep) =>
        lowerMatrixIR(child, ab).map {
          TableFilterIntervals(_, intervals, keep)
        }

      case MatrixUnionRows(children) =>
        // FIXME: this should check that all children have the same column keys.
        children.traverse(lowerMatrixIR(_, ab)).map { newChildren =>
          TableUnion(newChildren)
            .mapRows('row.selectFields(newChildren.head.typ.rowType.fieldNames: _*))
        }

      case MatrixDistinctByRow(child) =>
        lowerMatrixIR(child, ab).map(TableDistinct)

      case MatrixRowsHead(child, n) =>
        lowerMatrixIR(child, ab).map(TableHead(_, n))

      case MatrixRowsTail(child, n) =>
        lowerMatrixIR(child, ab).map(TableTail(_, n))

      case MatrixColsHead(child, n) =>
        lowerMatrixIR(child, ab).map(_
          .mapGlobals('global.insertFields(colsField -> 'global(colsField).arraySlice(0, Some(n), 1)))
          .mapRows('row.insertFields(entriesField -> 'row(entriesField).arraySlice(0, Some(n), 1)))
        )

      case MatrixColsTail(child, n) =>
        lowerMatrixIR(child, ab).map(_
          .mapGlobals('global.insertFields(colsField -> 'global(colsField).arraySlice(-n, None, 1)))
          .mapRows('row.insertFields(entriesField -> 'row(entriesField).arraySlice(-n, None, 1)))
        )

      case MatrixExplodeCols(child, path) =>
        lowerMatrixIR(child, ab).map { loweredChild =>
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
            .mapGlobals('global.insertFields(lengths -> 'global(colsField).map({
              colElementUID1 ~> arrayIR.len.orElse(0)
            })))
            .mapGlobals('global.insertFields(colsField ->
              irRange(0, 'global(colsField).len, 1)
                .flatMap({
                  colIdx ~>
                    irRange(0, 'global(lengths)(colIdx), 1)
                      .map({
                        nestedIdx ~> postExplodeSelector
                      })
                })))
            .mapRows('row.insertFields(entriesField ->
              irRange(0, 'row(entriesField).len, 1)
                .flatMap(colIdx ~>
                  irRange(0, 'global(lengths)(colIdx), 1).map(Symbol(genUID()) ~> 'row(entriesField)(colIdx)))))
            .mapGlobals('global.dropFields(lengths))
        }

      case MatrixAggregateRowsByKey(child, entryExpr, rowExpr) =>
        for {
          loweredChild <- lowerMatrixIR(child, ab)
          loweredEntry <- lowerIR(entryExpr, ab)
          loweredRow <- lowerIR(rowExpr, ab)
        } yield {
          val substEnv = matrixSubstEnv(child)
          val eeSub = subst(loweredEntry, substEnv)
          val reSub = subst(loweredRow, substEnv)
          loweredChild
            .aggregateByKey(
              reSub.insertFields(entriesField -> irRange(0, 'global(colsField).len)
                .aggElements('__element_idx, '__result_idx, Some('global(colsField).len))(
                  let(sa = 'global(colsField)('__result_idx)) {
                    aggLet(sa = 'global(colsField)('__element_idx),
                      g = 'row(entriesField)('__element_idx)) {
                      aggFilter(!'g.isNA, eeSub)
                    }
                  })))
        }

      case MatrixCollectColsByKey(child) =>
        lowerMatrixIR(child, ab).map(_
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
        )

      case MatrixExplodeRows(child, path) =>
        lowerMatrixIR(child, ab).map {
          TableExplode(_, path)
        }

      case mr: MatrixRead =>
        mr.lower[M]

      case MatrixAggregateColsByKey(child, entryExpr, colExpr) =>
        for {
          loweredChild <- lowerMatrixIR(child, ab)
          loweredEntry <- lowerIR(entryExpr, ab)
          loweredCol <- lowerIR(colExpr, ab)
        } yield {
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
          val ceSub = subst(loweredCol, BindingEnv(e2, agg = Some(e1)))
          val eeSub = subst(loweredEntry, BindingEnv(e1, agg = Some(e1)))

          loweredChild
            .mapGlobals('global.insertFields(keyMap ->
              let(__cols_field = 'global(colsField)) {
                irRange(0, '__cols_field.len)
                  .map(originalColIdx ~> let(__cols_field_element = '__cols_field(originalColIdx)) {
                    makeStruct('key -> '__cols_field_element.selectFields(colKey: _*), 'value -> originalColIdx)
                  })
                  .groupByKey
                  .toArray
              }))
            .mapRows('row.insertFields(entriesField ->
              let(__entries = 'row(entriesField), __key_map = 'global(keyMap)) {
                irRange(0, '__key_map.len)
                  .map(newColIdx1 ~> '__key_map(newColIdx1)
                    .apply('value)
                    .streamAgg(aggElementIdx ~>
                      aggLet(g = '__entries(aggElementIdx), sa = 'global(colsField)(aggElementIdx)) {
                        aggFilter(!'g.isNA, eeSub)
                      }))
              }))
            .mapGlobals(
              'global.insertFields(colsField ->
                let(__key_map = 'global(keyMap)) {
                  irRange(0, '__key_map.len)
                    .map(newColIdx2 ~>
                      concatStructs(
                        '__key_map(newColIdx2)('key),
                        '__key_map(newColIdx2)('value)
                          .streamAgg(colsAggIdx ~> aggLet(sa = 'global(colsField)(colsAggIdx)) {
                            ceSub
                          })
                      ))
                }
              ).dropFields(keyMap))
        }

      case MatrixLiteral(_, tl) =>
        Applicative[M].pure(tl)
    }

    lowered.map { lowered =>
      if (!mir.typ.isCompatibleWith(lowered.typ))
        throw new RuntimeException(s"Lowering changed type:\n  BEFORE: ${Pretty(mir)}\n    ${mir.typ}\n    ${mir.typ.canonicalTableType}\n  AFTER: ${Pretty(lowered)}\n    ${lowered.typ}")
      lowered
    }
  }

  private[this] def lowerTableIR[M[_]: MonadLower](tir: TableIR, ab: BoxedArrayBuilder[(String, IR)])
  : M[TableIR] =
    assertTypeUnchanged(tir) {
      tir match {
        case CastMatrixToTable(child, entries, cols) =>
          lowerMatrixIR(child, ab).map(_
            .mapRows('row.selectFields(child.typ.rowType.fieldNames ++ Array(entriesFieldName): _*))
            .mapGlobals('global.selectFields(child.typ.globalType.fieldNames ++ Array(colsFieldName): _*))
            .rename(Map(entriesFieldName -> entries), Map(colsFieldName -> cols))
          )

        case x@MatrixEntriesTable(child) =>
          lowerMatrixIR(child, ab).map { lc =>

            if (child.typ.rowKey.nonEmpty && child.typ.colKey.nonEmpty) {
              val oldColIdx = Symbol(genUID())
              val lambdaIdx1 = Symbol(genUID())
              val lambdaIdx2 = Symbol(genUID())
              val lambdaIdx3 = Symbol(genUID())
              val toExplode = Symbol(genUID())
              val values = Symbol(genUID())
              lc.mapGlobals('global.insertFields(oldColIdx -> irRange(0, 'global(colsField).len)
                .map(lambdaIdx1 ~> makeStruct('key -> 'global(colsField)(lambdaIdx1).selectFields(child.typ.colKey: _*), 'value -> lambdaIdx1))
                .sort(ascending = true, onKey = true)
                .map(lambdaIdx1 ~> lambdaIdx1('value))))
                .aggregateByKey(makeStruct(values -> applyAggOp(Collect(), seqOpArgs = FastIndexedSeq('row.selectFields(lc.typ.valueType.fieldNames: _*)))))
                .mapRows('row.dropFields(values).insertFields(toExplode ->
                  'global(oldColIdx)
                    .flatMap(lambdaIdx1 ~> 'row(values)
                      .filter(lambdaIdx2 ~> !lambdaIdx2(entriesField)(lambdaIdx1).isNA)
                      .map(lambdaIdx3 ~> let(__col = 'global(colsField)(lambdaIdx1), __entry = lambdaIdx3(entriesField)(lambdaIdx1)) {
                        makeStruct(
                          child.typ.rowValueStruct.fieldNames.map(Symbol(_)).map(f => f -> lambdaIdx3(f)) ++
                            child.typ.colType.fieldNames.map(Symbol(_)).map(f => f -> '__col(f)) ++
                            child.typ.entryType.fieldNames.map(Symbol(_)).map(f => f -> '__entry(f)): _*
                        )
                      }
                      )
                    )
                ))
                .explode(toExplode)
                .mapRows(makeStruct(x.typ.rowType.fieldNames.map { f =>
                  val fd = Symbol(f)
                  (fd, if (child.typ.rowKey.contains(f)) 'row(fd) else 'row(toExplode)(fd))
                }: _*))
                .mapGlobals('global.dropFields(colsField, oldColIdx))
                .keyBy(child.typ.rowKey ++ child.typ.colKey, isSorted = true)
            } else {
              val colIdx = Symbol(genUID())
              val lambdaIdx = Symbol(genUID())
              val result = lc
                .mapRows('row.insertFields(colIdx -> irRange(0, 'global(colsField).len)
                  .filter(lambdaIdx ~> !'row(entriesField)(lambdaIdx).isNA)))
                .explode(colIdx)
                .mapRows(let(__col_struct = 'global(colsField)('row(colIdx)),
                  __entry_struct = 'row(entriesField)('row(colIdx))) {
                  val newFields = child.typ.colType.fieldNames.map(Symbol(_)).map(f => f -> '__col_struct(f)) ++
                    child.typ.entryType.fieldNames.map(Symbol(_)).map(f => f -> '__entry_struct(f))

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
          }

        case MatrixToTableApply(child, function) =>
          lowerMatrixIR(child, ab).map {
            TableToTableApply(_,
              function.lower().getOrElse(
                WrappedMatrixToTableFunction(function, colsFieldName, entriesFieldName, child.typ.colKey)
              )
            )
          }

        case MatrixRowsTable(child) =>
          lowerMatrixIR(child, ab).map {
            _.mapGlobals('global.dropFields(colsField))
              .mapRows('row.dropFields(entriesField))
          }

        case MatrixColsTable(child) =>
          val colKey = child.typ.colKey
          lowerMatrixIR(child, ab).map { lowered =>
            let(__cols_and_globals = lowered.getGlobals) {
              val sortedCols = if (colKey.isEmpty)
                '__cols_and_globals(colsField)
              else
                '__cols_and_globals(colsField).map {
                  '__cols_element ~>
                    makeStruct(
                      // key struct
                      '_1 -> '__cols_element.selectFields(colKey: _*),
                      '_2 -> '__cols_element)
                }.sort(true, onKey = true)
                  .map {
                    'elt ~> 'elt('_2)
                  }
              makeStruct('rows -> sortedCols, 'global -> '__cols_and_globals.dropFields(colsField))
            }.parallelize(None).keyBy(child.typ.colKey)
          }

        case table => lowerChildren(table, ab).map(_.asInstanceOf[TableIR])
      }
    }

  private[this] def lowerBlockMatrixIR[M[_]: MonadLower](bmir: BlockMatrixIR,
                                                         ab: BoxedArrayBuilder[(String, IR)]
                                                        ): M[BlockMatrixIR] =
    assertTypeUnchanged(bmir) {
      for {ir <- lowerChildren(bmir, ab)}
        yield ir.asInstanceOf[BlockMatrixIR]
    }

  private[this] def lowerIR[M[_]: MonadLower](ir: IR, ab: BoxedArrayBuilder[(String, IR)]): M[IR] =
    assertTypeUnchanged(ir) {
      ir match {
        case MatrixToValueApply(child, function) =>
          lowerMatrixIR(child, ab).map { tableIr =>
            TableToValueApply(
              tableIr,
              function.lower().getOrElse(
                WrappedMatrixToValueFunction(function, colsFieldName, entriesFieldName, child.typ.colKey)
              )
            )
          }

        case MatrixWrite(child, writer) =>
          lowerMatrixIR(child, ab).map { tir =>
            TableWrite(tir, WrappedMatrixWriter(writer, colsFieldName, entriesFieldName, child.typ.colKey))
          }

        case MatrixMultiWrite(children, writer) =>
          children.traverse(lowerMatrixIR(_, ab)).map { lowered =>
            TableMultiWrite(lowered, WrappedMatrixNativeMultiWriter(writer, children.head.typ.colKey))
          }

        case MatrixCount(child) =>
          lowerMatrixIR(child, ab).map {
            _.aggregate(makeTuple(applyAggOp(Count(), FastIndexedSeq(), FastIndexedSeq()), 'global(colsField).len))
          }

        case MatrixAggregate(child, query) =>
          lowerMatrixIR(child, ab).map { lc =>
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
                  ApplyUnaryPrimOp(Bang(), IsNA(GetTupleElement(filterTuple, 0)))
                }) { explodedTuple =>
                AggLet("g", GetTupleElement(explodedTuple, 0),
                  AggLet("sa", GetTupleElement(explodedTuple, 1), Subst(query, matrixSubstEnvIR(child, lc)),
                    isScan = false),
                  isScan = false
                )
              })
          }
        case _ => lowerChildren(ir, ab).map(_.asInstanceOf[IR])
      }
    }

  private[this] def assertTypeUnchanged[M[_], A <: BaseIR](original: A)(lowered: M[A])
                                                          (implicit M: MonadThrow[M]): M[A] =
    for {
      l <- lowered
      _ <- M.raiseWhen(l.typ != original.typ) {
        new HailException(
          s"lowering changed type:" +
            s"\n  before: ${original.typ}" +
            s"\n  after: ${l.typ}" +
            s"\n  ${ original.getClass.getName } => ${ lowered.getClass.getName })"
        )
      }
    } yield l
}

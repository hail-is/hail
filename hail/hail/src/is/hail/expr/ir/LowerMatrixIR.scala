package is.hail.expr.ir

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.defs._
import is.hail.expr.ir.functions.{WrappedMatrixToTableFunction, WrappedMatrixToValueFunction}
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.utils.compat.immutable.ArraySeq
import is.hail.utils.compat.mutable.Growable

object LowerMatrixIR {
  val entriesFieldName: String = MatrixType.entriesIdentifier
  val colsFieldName: String = "__cols"
  val colsField: Symbol = Symbol(colsFieldName)
  val entriesField: Symbol = Symbol(entriesFieldName)

  def apply(ctx: ExecuteContext, ir: IR): IR = {
    val ab = ArraySeq.newBuilder[(Name, IR)]
    val l1 = lower(ctx, ir, ab)
    ab.result().foldRight[IR](l1) { case ((ident, value), body) =>
      RelationalLet(ident, value, body)
    }
  }

  def apply(ctx: ExecuteContext, tir: TableIR): TableIR = {
    val ab = ArraySeq.newBuilder[(Name, IR)]
    val l1 = lower(ctx, tir, ab)
    ab.result().foldRight[TableIR](l1) { case ((ident, value), body) =>
      RelationalLetTable(ident, value, body)
    }
  }

  def apply(ctx: ExecuteContext, mir: MatrixIR): TableIR = {
    val ab = ArraySeq.newBuilder[(Name, IR)]

    val l1 = lower(ctx, mir, ab)
    ab.result().foldRight[TableIR](l1) { case ((ident, value), body) =>
      RelationalLetTable(ident, value, body)
    }
  }

  def apply(ctx: ExecuteContext, bmir: BlockMatrixIR): BlockMatrixIR = {
    val ab = ArraySeq.newBuilder[(Name, IR)]

    val l1 = lower(ctx, bmir, ab)
    assert(ab.result().isEmpty)
    l1
  }

  private[this] def lowerChildren(
    ctx: ExecuteContext,
    ir: BaseIR,
    ab: Growable[(Name, IR)],
  ): BaseIR = {
    ir.mapChildren {
      case tir: TableIR => lower(ctx, tir, ab)
      case mir: MatrixIR => throw new RuntimeException(s"expect specialized lowering rule for " +
          s"${ir.getClass.getName}\n  Found MatrixIR child $mir")
      case bmir: BlockMatrixIR => lower(ctx, bmir, ab)
      case vir: IR => lower(ctx, vir, ab)
    }
  }

  def colVals(tir: TableIR): IR =
    GetField(Ref(TableIR.globalName, tir.typ.globalType), colsFieldName)

  def globals(tir: TableIR): IR =
    SelectFields(
      Ref(TableIR.globalName, tir.typ.globalType),
      tir.typ.globalType.fieldNames.diff(FastSeq(colsFieldName)),
    )

  def nCols(tir: TableIR): IR = ArrayLen(colVals(tir))

  def entries(tir: TableIR): IR =
    GetField(Ref(TableIR.rowName, tir.typ.rowType), entriesFieldName)

  import is.hail.expr.ir.DeprecatedIRBuilder._

  def matrixSubstEnv(child: MatrixIR): BindingEnv[IRProxy] = {
    val e = Env[IRProxy](
      MatrixIR.globalName -> 'global.selectFields(child.typ.globalType.fieldNames: _*),
      MatrixIR.rowName -> 'row.selectFields(child.typ.rowType.fieldNames: _*),
    )
    BindingEnv(e, agg = Some(e), scan = Some(e))
  }

  def matrixGlobalSubstEnv(child: MatrixIR): BindingEnv[IRProxy] = {
    val e =
      Env[IRProxy](MatrixIR.globalName -> 'global.selectFields(child.typ.globalType.fieldNames: _*))
    BindingEnv(e, agg = Some(e), scan = Some(e))
  }

  def matrixSubstEnvIR(child: MatrixIR, lowered: TableIR): BindingEnv[IR] = {
    val e = Env[IR](
      MatrixIR.globalName -> SelectFields(
        Ref(TableIR.globalName, lowered.typ.globalType),
        child.typ.globalType.fieldNames,
      ),
      MatrixIR.rowName -> SelectFields(
        Ref(TableIR.rowName, lowered.typ.rowType),
        child.typ.rowType.fieldNames,
      ),
    )
    BindingEnv(e, agg = Some(e), scan = Some(e))
  }

  private def bindingsToStruct(bindings: IndexedSeq[(Name, IR)]): MakeStruct =
    MakeStruct(bindings.map { case (n, ir) => n.str -> ir })

  private def unwrapStruct(bindings: IndexedSeq[(Name, IR)], struct: Ref): IndexedSeq[(Name, IR)] =
    bindings.map { case (name, _) => name -> GetField(struct, name.str) }

  private[this] def lower(
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
        val row = Ref(TableIR.rowName, lc.typ.rowType)
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
                        Str(
                          "length mismatch between entry array and column array in 'to_matrix_table_row_major': "
                        ),
                        invoke("str", TString, entriesLen),
                        Str(" entries, "),
                        invoke("str", TString, colsLen),
                        Str(" cols, at "),
                        invoke("str", TString, SelectFields(row, child.typ.key)),
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
          t = t.mapGlobals('global.castRename(t.typ.globalType.insertFields(FastSeq((
            colsFieldName,
            newColsType,
          )))))
        }

        if (entryMap.nonEmpty) {
          val newEntriesType = TArray(child.typ.entryType.rename(entryMap))
          t = t.mapRows('row.castRename(t.typ.rowType.insertFields(FastSeq((
            entriesFieldName,
            newEntriesType,
          )))))
        }

        t

      case MatrixKeyRowsBy(child, keys, isSorted) =>
        lower(ctx, child, liftedRelationalLets).keyBy(keys, isSorted)

      case MatrixFilterRows(child, pred) =>
        lower(ctx, child, liftedRelationalLets)
          .filter(subst(lower(ctx, pred, liftedRelationalLets), matrixSubstEnv(child)))

      case MatrixFilterCols(child, pred) =>
        lower(ctx, child, liftedRelationalLets)
          .mapGlobals('global.insertFields('newColIdx ->
            irRange(0, 'global(colsField).len)
              .filter('i ~>
                (let(sa = 'global(colsField)('i))
                  in subst(lower(ctx, pred, liftedRelationalLets), matrixGlobalSubstEnv(child))))))
          .mapRows('row.insertFields(
            entriesField -> 'global('newColIdx).map('i ~> 'row(entriesField)('i))
          ))
          .mapGlobals('global
            .insertFields(colsField ->
              'global('newColIdx).map('i ~> 'global(colsField)('i)))
            .dropFields('newColIdx))

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
          .mapGlobals('global.insertFields('newColIdx -> oldIndices.map(I32)))
          .mapRows('row.insertFields(
            entriesField -> 'global('newColIdx).map('i ~> 'row(entriesField)('i))
          ))
          .mapGlobals('global
            .insertFields(colsField -> 'global('newColIdx).map('i ~> 'global(colsField)('i)))
            .dropFields('newColIdx))

      case MatrixAnnotateColsTable(child, table, root) =>
        val col = Symbol(genUID())
        val colKey = makeStruct(table.typ.key.zip(child.typ.colKey).map { case (tk, mck) =>
          Symbol(tk) -> col(Symbol(mck))
        }: _*)
        lower(ctx, child, liftedRelationalLets)
          .mapGlobals(let(__dictfield =
            lower(ctx, table, liftedRelationalLets)
              .keyBy(FastSeq())
              .collect()
              .apply('rows)
              .arrayStructToDict(table.typ.key)
          ) {
            'global.insertFields(colsField ->
              'global(colsField).map(col ~> col.insertFields(Symbol(root) -> '__dictfield.invoke(
                "get",
                table.typ.valueType,
                colKey,
              ))))
          })

      case MatrixMapGlobals(child, newGlobals) =>
        lower(ctx, child, liftedRelationalLets)
          .mapGlobals(
            subst(
              lower(ctx, newGlobals, liftedRelationalLets),
              BindingEnv(Env[IRProxy](
                TableIR.globalName -> 'global.selectFields(child.typ.globalType.fieldNames: _*)
              )),
            )
              .insertFields(colsField -> 'global(colsField))
          )

      case MatrixMapRows(child, newRow) =>
        def liftScans(ir: IR): IRProxy = {
          def lift(ir: IR, builder: Growable[(Name, IR)]): IR = ir match {
            case a: ApplyScanOp =>
              val s = freshName()
              builder += ((s, a))
              Ref(s, a.typ)

            case a @ AggFold(_, _, _, _, _, true) =>
              val s = freshName()
              builder += ((s, a))
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
              val elementType = aggIR.typ.asInstanceOf[TDict].elementType
              val valueType = elementType.types(1)
              val valueUID = Ref(freshName(), valueType)
              ToDict(mapIR(ToStream(uid)) { eltUID =>
                Let(
                  (valueUID.name -> GetField(eltUID, "value")) +: unwrapStruct(aggs, valueUID),
                  MakeTuple.ordered(FastSeq(GetField(eltUID, "key"), liftedBody)),
                )
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
                if (i == bindings.length) {
                  lift(body, builder)
                } else bindings(i) match {
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

          val scans = ab.result().toFastSeq
          val scanStruct = MakeStruct(scans.map { case (n, ir) => n.str -> ir })

          val scanResultRef = Ref(freshName(), scanStruct.typ)

          val b1 = if (ContainsAgg(b0)) {
            irRange(0, 'row(entriesField).len)
              .filter('i ~> !'row(entriesField)('i).isNA)
              .streamAgg('i ~>
                (aggLet(sa = 'global(colsField)('i), g = 'row(entriesField)('i))
                  in b0))
          } else
            irToProxy(b0)

          letDyn(
            ((scanResultRef.name, irToProxy(scanStruct))
              +: scans.map { case (name, _) =>
                name -> irToProxy(GetField(scanResultRef, name.str))
              }): _*
          )(b1)
        }

        val lc = lower(ctx, child, liftedRelationalLets)
        lc.mapRows(let(n_cols = 'global(colsField).len) {
          liftScans(Subst(lower(ctx, newRow, liftedRelationalLets), matrixSubstEnvIR(child, lc)))
            .insertFields(entriesField -> 'row(entriesField))
        })

      case MatrixMapCols(child, newCol, _) =>
        val loweredChild = lower(ctx, child, liftedRelationalLets)

        def lift(ir: IR, scanBindings: Growable[(Name, IR)], aggBindings: Growable[(Name, IR)])
          : IR = ir match {
          case a: ApplyScanOp =>
            val s = freshName()
            scanBindings += ((s, a))
            Ref(s, a.typ)

          case a: ApplyAggOp =>
            val s = freshName()
            aggBindings += ((s, a))
            Ref(s, a.typ)

          case a @ AggFold(_, _, _, _, _, isScan) =>
            val s = freshName()
            if (isScan) {
              scanBindings += ((s, a))
            } else {
              aggBindings += ((s, a))
            }
            Ref(s, a.typ)

          case AggFilter(filt, body, isScan) =>
            val ab = ArraySeq.newBuilder[(Name, IR)]
            val (liftedBody, builder) =
              if (isScan) (lift(body, ab, aggBindings), scanBindings)
              else (lift(body, scanBindings, ab), aggBindings)

            val aggs = ab.result()
            val structResult = MakeStruct(aggs.map { case (n, ir) => n.str -> ir })

            val uid = Ref(freshName(), structResult.typ)
            builder += (uid.name -> AggFilter(filt, structResult, isScan))
            Let(aggs.map { case (name, _) => name -> GetField(uid, name.str) }, liftedBody)

          case AggExplode(a, name, body, isScan) =>
            val ab = ArraySeq.newBuilder[(Name, IR)]
            val (liftedBody, builder) =
              if (isScan) (lift(body, ab, aggBindings), scanBindings)
              else (lift(body, scanBindings, ab), aggBindings)

            val aggs = ab.result()
            val structResult = MakeStruct(aggs.map { case (n, ir) => n.str -> ir })
            val uid = Ref(freshName(), structResult.typ)
            builder += (uid.name -> AggExplode(a, name, structResult, isScan))
            Let(aggs.map { case (name, _) => name -> GetField(uid, name.str) }, liftedBody)

          case AggGroupBy(a, body, isScan) =>
            val ab = ArraySeq.newBuilder[(Name, IR)]
            val (liftedBody, builder) =
              if (isScan) (lift(body, ab, aggBindings), scanBindings)
              else (lift(body, scanBindings, ab), aggBindings)

            val aggs = ab.result()
            val aggIR = AggGroupBy(a, MakeStruct(aggs.map { case (n, ir) => n.str -> ir }), isScan)
            val uid = Ref(freshName(), aggIR.typ)
            builder += (uid.name -> aggIR)
            val valueUID = freshName()
            val elementType = aggIR.typ.asInstanceOf[TDict].elementType
            val valueType = elementType.types(1)
            ToDict(mapIR(ToStream(uid)) { eltUID =>
              MakeTuple.ordered(
                FastSeq(
                  GetField(eltUID, "key"),
                  Let(
                    (valueUID -> GetField(eltUID, "value")) +:
                      aggs.map { case (name, _) =>
                        name -> GetField(Ref(valueUID, valueType), name.str)
                      },
                    liftedBody,
                  ),
                )
              )
            })

          case AggArrayPerElement(a, elementName, indexName, body, knownLength, isScan) =>
            val ab = ArraySeq.newBuilder[(Name, IR)]
            val (liftedBody, builder) =
              if (isScan) (lift(body, ab, aggBindings), scanBindings)
              else (lift(body, scanBindings, ab), aggBindings)

            val aggs = ab.result()
            val aggBody = MakeStruct(aggs.map { case (n, ir) => n.str -> ir })
            val aggIR =
              AggArrayPerElement(a, elementName, indexName, aggBody, knownLength, isScan)
            val uid = Ref(freshName(), aggIR.typ)
            builder += (uid.name -> aggIR)
            ToArray(mapIR(ToStream(uid)) { eltUID =>
              Let(aggs.map { case (name, _) => name -> GetField(eltUID, name.str) }, liftedBody)
            })

          case Block(bindings, body) =>
            var newBindings = Seq[Binding]()
            def go(i: Int, scanBindings: Growable[(Name, IR)], aggBindings: Growable[(Name, IR)])
              : IR = {
              if (i == bindings.length) {
                lift(body, scanBindings, aggBindings)
              } else bindings(i) match {
                case Binding(name, value, Scope.EVAL) =>
                  val lifted = lift(value, scanBindings, aggBindings)
                  val liftedBody = go(i + 1, scanBindings, aggBindings)
                  newBindings = Binding(name, lifted, Scope.EVAL) +: newBindings
                  liftedBody
                case Binding(name, value, scope) =>
                  val ab = ArraySeq.newBuilder[(Name, IR)]
                  val liftedBody = if (scope == Scope.SCAN)
                    go(i + 1, ab, aggBindings)
                  else
                    go(i + 1, scanBindings, ab)

                  val builder = if (scope == Scope.SCAN) scanBindings else aggBindings

                  val aggs = ab.result()
                  val structResult = MakeStruct(aggs.map { case (n, ir) => n.str -> ir })

                  val uid = freshName()
                  builder += (uid -> Block(FastSeq(Binding(name, value, scope)), structResult))
                  newBindings = aggs.map { case (name, _) =>
                    Binding(name, GetField(Ref(uid, structResult.typ), name.str), Scope.EVAL)
                  } ++ newBindings
                  liftedBody
              }
            }
            val newBody = go(0, scanBindings, aggBindings)
            Block(newBindings.toFastSeq, newBody)

          case x: StreamAgg => x
          case x: StreamAggScan => x

          case _ =>
            MapIR(lift(_, scanBindings, aggBindings))(ir)
        }

        val scanBuilder = ArraySeq.newBuilder[(Name, IR)]
        val aggBuilder = ArraySeq.newBuilder[(Name, IR)]

        val b0 = lift(
          Subst(lower(ctx, newCol, liftedRelationalLets), matrixSubstEnvIR(child, loweredChild)),
          scanBuilder,
          aggBuilder,
        )
        val aggs = aggBuilder.result()
        val scans = scanBuilder.result()

        val idx = Ref(freshName(), TInt32)
        val idxSym = Symbol(idx.name.str)

        val noOp: (IRProxy => IRProxy, IRProxy => IRProxy) = (identity[IRProxy], identity[IRProxy])

        val (
          aggOutsideTransformer: (IRProxy => IRProxy),
          aggInsideTransformer: (IRProxy => IRProxy),
        ) = if (aggs.isEmpty)
          noOp
        else {
          val aggStruct = MakeStruct(aggs.map { case (n, ir) => n.str -> ir })

          val aggResult = loweredChild.aggregate(
            aggLet(va = 'row.selectFields(child.typ.rowType.fieldNames: _*)) {
              makeStruct(
                ('count, applyAggOp(Count(), FastSeq(), FastSeq())),
                (
                  'array_aggs,
                  irRange(0, 'global(colsField).len)
                    .aggElements('__element_idx, '__result_idx, Some('global(colsField).len))(
                      let(sa = 'global(colsField)('__result_idx)) {
                        aggLet(
                          sa = 'global(colsField)('__element_idx),
                          g = 'row(entriesField)('__element_idx),
                        ) {
                          aggFilter(!'g.isNA, aggStruct)
                        }
                      }
                    ),
                ),
              )
            }
          )

          val ident = freshName()
          liftedRelationalLets += ((ident, aggResult))

          val aggResultRef = Ref(freshName(), aggResult.typ)
          val aggResultElementRef = Ref(
            freshName(),
            aggResult.typ.asInstanceOf[TStruct]
              .fieldType("array_aggs")
              .asInstanceOf[TArray].elementType,
          )

          val bindResult: IRProxy => IRProxy = letDyn((
            aggResultRef.name,
            irToProxy(RelationalRef(ident, aggResult.typ)),
          )).apply(_)
          val bodyResult: IRProxy => IRProxy = (x: IRProxy) =>
            letDyn((
              aggResultRef.name,
              irToProxy(RelationalRef(ident, aggResult.typ)),
            ))
              .apply(let(
                n_rows = Symbol(aggResultRef.name.str)('count),
                array_aggs = Symbol(aggResultRef.name.str)('array_aggs),
              ) {
                letDyn((aggResultElementRef.name, 'array_aggs(idx))) {
                  aggs.foldLeft[IRProxy](x) { case (acc, (name, _)) =>
                    letDyn((name, GetField(aggResultElementRef, name.str)))(acc)
                  }
                }
              })
          (bindResult, bodyResult)
        }

        val (
          scanOutsideTransformer: (IRProxy => IRProxy),
          scanInsideTransformer: (IRProxy => IRProxy),
        ) = if (scans.isEmpty)
          noOp
        else {
          val scanStruct = bindingsToStruct(scans)

          val scanResultArray = ToArray(StreamAggScan(
            ToStream(GetField(Ref(TableIR.globalName, loweredChild.typ.globalType), colsFieldName)),
            MatrixIR.colName,
            scanStruct,
          ))

          val scanResultRef = Ref(freshName(), scanResultArray.typ)
          val scanResultElementRef =
            Ref(freshName(), scanResultArray.typ.asInstanceOf[TArray].elementType)

          val bindResult: IRProxy => IRProxy =
            letDyn((scanResultRef.name, scanResultArray)).apply(_)
          val bodyResult: IRProxy => IRProxy = (x: IRProxy) =>
            letDyn((
              scanResultElementRef.name,
              ArrayRef(scanResultRef, idx),
            ))(
              scans.foldLeft[IRProxy](x) { case (acc, (name, _)) =>
                letDyn((name, GetField(scanResultElementRef, name.str)))(acc)
              }
            )
          (bindResult, bodyResult)
        }

        loweredChild.mapGlobals('global.insertFields(colsField ->
          aggOutsideTransformer(scanOutsideTransformer(irRange(0, 'global(colsField).len).map(
            idxSym ~> let(__cols_array = 'global(colsField), sa = '__cols_array(idxSym)) {
              aggInsideTransformer(scanInsideTransformer(b0))
            }
          )))))

      case MatrixFilterEntries(child, pred) =>
        val lc = lower(ctx, child, liftedRelationalLets)
        lc.mapRows('row.insertFields(entriesField ->
          irRange(0, 'global(colsField).len).map {
            'i ~>
              let(g = 'row(entriesField)('i)) {
                irIf(let(sa = 'global(colsField)('i))
                  in !subst(lower(ctx, pred, liftedRelationalLets), matrixSubstEnv(child))) {
                  NA(child.typ.entryType)
                } {
                  'g
                }
              }
          }))

      case MatrixUnionCols(left, right, joinType) =>
        val rightEntries = genUID()
        val rightCols = genUID()
        val ll = lower(ctx, left, liftedRelationalLets).distinct()
        def handleMissingEntriesArray(entries: Symbol, cols: Symbol): IRProxy =
          if (joinType == "inner")
            'row(entries)
          else
            irIf('row(entries).isNA) {
              irRange(0, 'global(cols).len)
                .map('a ~> irToProxy(MakeStruct(right.typ.entryType.fieldNames.map(f =>
                  (f, NA(right.typ.entryType.fieldType(f)))
                ))))
            } {
              'row(entries)
            }
        val rr = lower(ctx, right, liftedRelationalLets).distinct()
        TableJoin(
          ll,
          rr.mapRows('row.castRename(rr.typ.rowType.rename(Map(entriesFieldName -> rightEntries))))
            .mapGlobals('global
              .insertFields(Symbol(rightCols) -> 'global(colsField))
              .selectFields(rightCols)),
          joinType,
        )
          .mapRows('row
            .insertFields(entriesField ->
              makeArray(
                handleMissingEntriesArray(entriesField, colsField),
                handleMissingEntriesArray(Symbol(rightEntries), Symbol(rightCols)),
              )
                .flatMap('a ~> 'a))
            .dropFields(Symbol(rightEntries)))
          .mapGlobals('global
            .insertFields(colsField ->
              makeArray('global(colsField), 'global(Symbol(rightCols))).flatMap('a ~> 'a))
            .dropFields(Symbol(rightCols)))

      case MatrixMapEntries(child, newEntries) =>
        val loweredChild = lower(ctx, child, liftedRelationalLets)
        val rt = loweredChild.typ.rowType
        val gt = loweredChild.typ.globalType
        TableMapRows(
          loweredChild,
          InsertFields(
            Ref(TableIR.rowName, rt),
            FastSeq((
              entriesFieldName,
              ToArray(StreamZip(
                FastSeq(
                  ToStream(GetField(Ref(TableIR.rowName, rt), entriesFieldName)),
                  ToStream(GetField(Ref(TableIR.globalName, gt), colsFieldName)),
                ),
                FastSeq(MatrixIR.entryName, MatrixIR.colName),
                Subst(
                  lower(ctx, newEntries, liftedRelationalLets),
                  BindingEnv(Env(
                    MatrixIR.globalName -> SelectFields(
                      Ref(TableIR.globalName, gt),
                      child.typ.globalType.fieldNames,
                    ),
                    MatrixIR.rowName -> SelectFields(
                      Ref(TableIR.rowName, rt),
                      child.typ.rowType.fieldNames,
                    ),
                  )),
                ),
                ArrayZipBehavior.AssumeSameLength,
              )),
            )),
          ),
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

      case MatrixColsHead(child, n) => lower(ctx, child, liftedRelationalLets)
          .mapGlobals('global.insertFields(colsField -> 'global(colsField).arraySlice(
            0,
            Some(n),
            1,
          )))
          .mapRows('row.insertFields(entriesField -> 'row(entriesField).arraySlice(0, Some(n), 1)))

      case MatrixColsTail(child, n) => lower(ctx, child, liftedRelationalLets)
          .mapGlobals('global.insertFields(colsField -> 'global(colsField).arraySlice(-n, None, 1)))
          .mapRows('row.insertFields(entriesField -> 'row(entriesField).arraySlice(-n, None, 1)))

      case MatrixExplodeCols(child, path) =>
        val loweredChild = lower(ctx, child, liftedRelationalLets)
        val lengths = Symbol(genUID())
        val colIdx = Symbol(genUID())
        val nestedIdx = Symbol(genUID())
        val colElementUID1 = Symbol(genUID())

        val nestedRefs =
          path.init.scanLeft('global(colsField)(colIdx): IRProxy)((irp, name) => irp(Symbol(name)))
        val postExplodeSelector = path.zip(nestedRefs).zipWithIndex.foldRight[IRProxy](nestedIdx) {
          case (((field, ref), i), arg) =>
            ref.insertFields(Symbol(field) ->
              (if (i == nestedRefs.length - 1)
                 ref(Symbol(field)).toArray(arg)
               else
                 arg))
        }

        val arrayIR = path.foldLeft[IRProxy](colElementUID1) { case (irp, fieldName) =>
          irp(Symbol(fieldName))
        }
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
                irRange(0, 'global(lengths)(colIdx), 1).map(
                  Symbol(genUID()) ~> 'row(entriesField)(colIdx)
                ))))
          .mapGlobals('global.dropFields(lengths))

      case MatrixAggregateRowsByKey(child, entryExpr, rowExpr) =>
        val substEnv = matrixSubstEnv(child)
        val eeSub = subst(lower(ctx, entryExpr, liftedRelationalLets), substEnv)
        val reSub = subst(lower(ctx, rowExpr, liftedRelationalLets), substEnv)
        lower(ctx, child, liftedRelationalLets)
          .aggregateByKey(
            reSub.insertFields(entriesField -> irRange(0, 'global(colsField).len)
              .aggElements('__element_idx, '__result_idx, Some('global(colsField).len))(
                let(sa = 'global(colsField)('__result_idx)) {
                  aggLet(
                    sa = 'global(colsField)('__element_idx),
                    g = 'row(entriesField)('__element_idx),
                  ) {
                    aggFilter(!'g.isNA, eeSub)
                  }
                }
              ))
          )

      case MatrixCollectColsByKey(child) =>
        lower(ctx, child, liftedRelationalLets)
          .mapGlobals('global.insertFields('newColIdx ->
            irRange(0, 'global(colsField).len).map {
              'i ~>
                makeTuple('global(colsField)('i).selectFields(child.typ.colKey: _*), 'i)
            }.groupByKey.toArray))
          .mapRows('row.insertFields(entriesField ->
            'global('newColIdx).map {
              'kv ~>
                makeStruct(child.typ.entryType.fieldNames.map { s =>
                  (
                    Symbol(s),
                    'kv('value).map {
                      'i ~> 'row(entriesField)('i)(Symbol(s))
                    },
                  )
                }: _*)
            }))
          .mapGlobals('global
            .insertFields(colsField ->
              'global('newColIdx).map {
                'kv ~>
                  'kv('key).insertFields(
                    child.typ.colValueStruct.fieldNames.map { s =>
                      (Symbol(s), 'kv('value).map('i ~> 'global(colsField)('i)(Symbol(s))))
                    }: _*
                  )
              })
            .dropFields('newColIdx))

      case MatrixExplodeRows(child, path) =>
        TableExplode(lower(ctx, child, liftedRelationalLets), path)

      case mr: MatrixRead => mr.lower(ctx)

      case MatrixAggregateColsByKey(child, entryExpr, colExpr) =>
        val colKey = child.typ.colKey

        val originalColIdx = Symbol(genUID())
        val newColIdx1 = Symbol(genUID())
        val newColIdx2 = Symbol(genUID())
        val colsAggIdx = Symbol(genUID())
        val keyMap = Symbol(genUID())
        val aggElementIdx = Symbol(genUID())

        val e1 = Env[IRProxy](
          MatrixIR.globalName -> 'global.selectFields(child.typ.globalType.fieldNames: _*),
          MatrixIR.rowName -> 'row.selectFields(child.typ.rowType.fieldNames: _*),
        )
        val e2 = Env[IRProxy](
          MatrixIR.globalName -> 'global.selectFields(child.typ.globalType.fieldNames: _*)
        )
        val ceSub =
          subst(lower(ctx, colExpr, liftedRelationalLets), BindingEnv(e2, agg = Some(e2)))
        val eeSub =
          subst(lower(ctx, entryExpr, liftedRelationalLets), BindingEnv(e1, agg = Some(e1)))

        lower(ctx, child, liftedRelationalLets)
          .mapGlobals('global.insertFields(keyMap ->
            let(__cols_field = 'global(colsField)) {
              irRange(0, '__cols_field.len)
                .map(originalColIdx ~> let(__cols_field_element = '__cols_field(originalColIdx)) {
                  makeStruct(
                    'key -> '__cols_field_element.selectFields(colKey: _*),
                    'value -> originalColIdx,
                  )
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
                        }),
                    ))
              }).dropFields(keyMap)
          )

      case MatrixLiteral(_, tl) => tl
    }

    if (!mir.typ.isCompatibleWith(lowered.typ))
      throw new RuntimeException(
        s"Lowering changed type:\n  BEFORE: ${Pretty(ctx, mir)}\n    ${mir.typ}\n    ${mir.typ.canonicalTableType}\n  AFTER: ${Pretty(ctx, lowered)}\n    ${lowered.typ}"
      )
    lowered
  }

  private[this] def lower(ctx: ExecuteContext, tir: TableIR, ab: Growable[(Name, IR)]): TableIR = {
    val lowered = tir match {
      case CastMatrixToTable(child, entries, cols) =>
        lower(ctx, child, ab)
          .mapRows('row.selectFields(child.typ.rowType.fieldNames ++ Array(entriesFieldName): _*))
          .mapGlobals('global.selectFields(
            child.typ.globalType.fieldNames ++ Array(colsFieldName): _*
          ))
          .rename(Map(entriesFieldName -> entries), Map(colsFieldName -> cols))

      case x @ MatrixEntriesTable(child) =>
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
              irRange(0, 'global(colsField).len)
                .map(lambdaIdx1 ~> makeStruct(
                  'key -> 'global(colsField)(lambdaIdx1).selectFields(child.typ.colKey: _*),
                  'value -> lambdaIdx1,
                ))
                .sort(ascending = true, onKey = true)
                .map(lambdaIdx1 ~> lambdaIdx1('value))))
            .aggregateByKey(makeStruct(values -> applyAggOp(
              Collect(),
              seqOpArgs = FastSeq('row.selectFields(lc.typ.valueType.fieldNames: _*)),
            )))
            .mapRows('row.dropFields(values).insertFields(toExplode ->
              'global(oldColIdx)
                .flatMap(lambdaIdx1 ~> 'row(values)
                  .filter(lambdaIdx2 ~> !lambdaIdx2(entriesField)(lambdaIdx1).isNA)
                  .map(lambdaIdx3 ~> let(
                    __col = 'global(colsField)(lambdaIdx1),
                    __entry = lambdaIdx3(entriesField)(lambdaIdx1),
                  ) {
                    makeStruct(
                      child.typ.rowValueStruct.fieldNames.map(Symbol(_)).map(f =>
                        f -> lambdaIdx3(f)
                      ) ++
                        child.typ.colType.fieldNames.map(Symbol(_)).map(f => f -> '__col(f)) ++
                        child.typ.entryType.fieldNames.map(Symbol(_)).map(f => f -> '__entry(f)): _*
                    )
                  }))))
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
            .mapRows(let(
              __col_struct = 'global(colsField)('row(colIdx)),
              __entry_struct = 'row(entriesField)('row(colIdx)),
            ) {
              val newFields =
                child.typ.colType.fieldNames.map(Symbol(_)).map(f => f -> '__col_struct(f)) ++
                  child.typ.entryType.fieldNames.map(Symbol(_)).map(f => f -> '__entry_struct(f))

              'row.dropFields(entriesField, colIdx).insertFieldsList(
                newFields,
                ordering = Some(x.typ.rowType.fieldNames.toFastSeq),
              )
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
        let(__cols_and_globals = lower(ctx, child, ab).getGlobals) {
          val sortedCols = if (colKey.isEmpty)
            '__cols_and_globals(colsField)
          else
            '__cols_and_globals(colsField).map {
              '__cols_element ~>
                makeStruct(
                  // key struct
                  '_1 -> '__cols_element.selectFields(colKey: _*),
                  '_2 -> '__cols_element,
                )
            }.sort(true, onKey = true)
              .map {
                'elt ~> 'elt('_2)
              }
          makeStruct('rows -> sortedCols, 'global -> '__cols_and_globals.dropFields(colsField))
        }.parallelize(None).keyBy(child.typ.colKey)

      case table => lowerChildren(ctx, table, ab).asInstanceOf[TableIR]
    }

    assertTypeUnchanged(tir, lowered)
    lowered
  }

  private[this] def lower(ctx: ExecuteContext, bmir: BlockMatrixIR, ab: Growable[(Name, IR)])
    : BlockMatrixIR = {
    val lowered = bmir match {
      case noMatrixChildren => lowerChildren(ctx, noMatrixChildren, ab).asInstanceOf[BlockMatrixIR]
    }
    assertTypeUnchanged(bmir, lowered)
    lowered
  }

  private[this] def lower(ctx: ExecuteContext, ir: IR, ab: Growable[(Name, IR)]): IR = {
    val lowered = ir match {
      case MatrixToValueApply(child, function) => TableToValueApply(
          lower(ctx, child, ab),
          function.lower()
            .getOrElse(WrappedMatrixToValueFunction(
              function,
              colsFieldName,
              entriesFieldName,
              child.typ.colKey,
            )),
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
          aggExplodeIR(
            filterIR(
              zip2(
                ToStream(GetField(Ref(TableIR.rowName, lc.typ.rowType), entriesFieldName)),
                ToStream(GetField(Ref(TableIR.globalName, lc.typ.globalType), colsFieldName)),
                ArrayZipBehavior.AssertSameLength,
              ) { case (e, c) =>
                MakeTuple.ordered(FastSeq(e, c))
              }
            )(filterTuple => ApplyUnaryPrimOp(Bang, IsNA(GetTupleElement(filterTuple, 0))))
          ) { explodedTuple =>
            AggLet(
              MatrixIR.entryName,
              GetTupleElement(explodedTuple, 0),
              AggLet(
                MatrixIR.colName,
                GetTupleElement(explodedTuple, 1),
                Subst(query, matrixSubstEnvIR(child, lc)),
                isScan = false,
              ),
              isScan = false,
            )
          },
        )
      case _ => lowerChildren(ctx, ir, ab).asInstanceOf[IR]
    }
    assertTypeUnchanged(ir, lowered)
    lowered
  }

  private[this] def assertTypeUnchanged(original: BaseIR, lowered: BaseIR): Unit =
    if (lowered.typ != original.typ)
      fatal(
        s"lowering changed type:\n  before: ${original.typ}\n after: ${lowered.typ}\n  ${original.getClass.getName} => ${lowered.getClass.getName}"
      )
}

package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.collection.compat.mutable.Growable
import is.hail.collection.implicits.toRichArray
import is.hail.expr.ir.{Memoized => M, _}
import is.hail.expr.ir.MatrixIR.{colName, entryName, globalName, rowName}
import is.hail.expr.ir.Scope._
import is.hail.expr.ir.defs._
import is.hail.expr.ir.functions.{WrappedMatrixToTableFunction, WrappedMatrixToValueFunction}
import is.hail.types.virtual._
import is.hail.utils._

object LowerMatrixIR extends Logging {
  val entriesFieldName: String = MatrixType.entriesIdentifier
  val colsFieldName: String = "__cols"

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

    if (lowered ne ir0) NormalizeNames()(ctx, lowered)
    else ir0
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

  private def bindingsToStruct(bindings: IndexedSeq[(Name, IR)]): MakeStruct =
    MakeStruct(bindings.map { case (n, ir) => n.str -> ir })

  private def unwrapStruct(bindings: IndexedSeq[(Name, _)], struct: Atom): IndexedSeq[(Name, IR)] =
    bindings.map { case (name, _) => name -> struct.get(name.str) }

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
        lower(ctx, child, liftedRelationalLets)
          .mapRows { (global, row) =>
            bindIR(row.get(entries)) { entries =>
              If(
                IsNA(entries),
                Die("missing entry array unsupported in 'to_matrix_table_row_major'", row.typ),
                bindIRs(entries.len, global.get(cols).len) {
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
                          row.select(child.typ.key),
                        ),
                        row.typ,
                        -1,
                      ),
                      row,
                    )
                },
              )
            }
          }
          .rename(Map(entries -> entriesFieldName), Map(cols -> colsFieldName))

      case MatrixToMatrixApply(child, function) =>
        val loweredChild = lower(ctx, child, liftedRelationalLets)
        TableToTableApply(loweredChild, function.lower())

      case MatrixRename(child, globalMap, colMap, rowMap, entryMap) =>
        var t = lower(ctx, child, liftedRelationalLets).rename(rowMap, globalMap)

        if (colMap.nonEmpty)
          t = t.mapGlobals { global =>
            global.rename(_.insertFields(FastSeq(
              colsFieldName -> TArray(child.typ.colType.rename(colMap))
            )))
          }

        if (entryMap.nonEmpty)
          t = t.mapRows { (_, row) =>
            row.rename(_.insertFields(FastSeq(
              entriesFieldName -> TArray(child.typ.entryType.rename(entryMap))
            )))
          }

        t

      case MatrixKeyRowsBy(child, keys, isSorted) =>
        lower(ctx, child, liftedRelationalLets).keyBy(keys, isSorted)

      case MatrixFilterRows(child, pred) =>
        lower(ctx, child, liftedRelationalLets).filter { (global, row) =>
          M.eval {
            M.sequence(
              globalName -> global.select(child.typ.globalType.fieldNames),
              rowName -> row.select(child.typ.rowType.fieldNames),
              lower(ctx, pred, liftedRelationalLets),
            )
          }
        }

      case MatrixFilterCols(child, pred) =>
        lower(ctx, child, liftedRelationalLets)
          .mapGlobals { global =>
            M.eval {
              for {
                cols <- global.get(colsFieldName)
                global <- Name("__global_save_filter_cols") -> global
                _ <- globalName -> global.select(child.typ.globalType.fieldNames)
              } yield global.insert(
                "__new_col_idx" ->
                  ToArray(rangeIR(cols.len).filter { idx =>
                    Let(
                      FastSeq(colName -> ArrayRef(cols, idx)),
                      lower(ctx, pred, liftedRelationalLets),
                    )
                  })
              )
            }
          }
          .mapRows { (global, row) =>
            row.update(entriesFieldName) { entries =>
              mapArray(global.get("__new_col_idx"))(entries.at(_))
            }
          }
          .mapGlobals { global =>
            global
              .update(colsFieldName)(cols => mapArray(global.get("__new_col_idx"))(cols.at(_)))
              .drop("__new_col_idx")
          }

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
          .mapGlobals(_.insert("__new_col_idx" -> Literal(TArray(TInt32), oldIndices)))
          .mapRows { (global, row) =>
            row.update(entriesFieldName) { entries =>
              mapArray(global.get("__new_col_idx"))(entries.at(_))
            }
          }
          .mapGlobals { global =>
            global
              .update(colsFieldName)(cols => mapArray(global.get("__new_col_idx"))(cols.at(_)))
              .drop("__new_col_idx")
          }

      case MatrixAnnotateColsTable(child, table, root) =>
        lower(ctx, child, liftedRelationalLets).mapGlobals { global =>
          bindIR(lower(ctx, table, liftedRelationalLets).collectAsDict) { annotations =>
            global.update(colsFieldName) { cols =>
              mapArray(cols) { col =>
                val key =
                  MakeStruct(
                    table.typ.key.zip(child.typ.colKey).map { case (tk, mck) =>
                      tk -> col.get(mck)
                    }
                  )

                col.insert(root -> annotations.invoke("get", table.typ.valueType, key))
              }
            }
          }
        }

      case MatrixMapGlobals(child, newGlobals) =>
        lower(ctx, child, liftedRelationalLets).mapGlobals { global =>
          M.eval {
            for {
              cols <- global.get(colsFieldName)
              _ <- globalName -> global.select(child.typ.globalType.fieldNames)
              newGlobals <- lower(ctx, newGlobals, liftedRelationalLets)
            } yield newGlobals.insert(colsFieldName -> cols)
          }
        }

      case MatrixMapRows(child, newRow) =>
        def liftScans(ir: IR): (IR, IndexedSeq[(Name, IR)]) = {
          def lift(ir: IR, builder: Growable[(Name, IR)]): IR =
            ir match {
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

                ToDict(mapIR(ToStream(uid)) { elt =>
                  M.eval {
                    for {
                      key <- elt.get("key")
                      value <- elt.get("value")
                      _ <- M.sequence(unwrapStruct(aggs, value).map(M.let[EVAL.type]): _*)
                    } yield maketuple(key, liftedBody)
                  }
                })

              case AggArrayPerElement(a, elementName, indexName, body, knownLength, true) =>
                val ab = ArraySeq.newBuilder[(Name, IR)]
                val liftedBody = lift(body, ab)

                val aggs = ab.result()
                val aggBody = bindingsToStruct(aggs)
                val aggIR =
                  AggArrayPerElement(a, elementName, indexName, aggBody, knownLength, true)
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
          (lift(ir, ab), ab.result())
        }

        lower(ctx, child, liftedRelationalLets).mapRows { (global, row) =>
          M.eval {
            for {
              entries <- row.get(entriesFieldName)
              cols <- global.get(colsFieldName)
              _ <- Name("n_cols") -> cols.len
              _ <- globalName -> global.select(child.typ.globalType.fieldNames)
              _ <- rowName -> row.select(child.typ.rowType.fieldNames)

              _ <- M.lift[SCAN.type] {
                M.sequence(
                  globalName -> global.select(child.typ.globalType.fieldNames),
                  rowName -> row.select(child.typ.rowType.fieldNames),
                  M.unit,
                )
              }

              (body, bindings) = liftScans(lower(ctx, newRow, liftedRelationalLets))

              _ <- M.sequence(bindings.map(M.let[EVAL.type]): _*)

              newRow <-
                if (!ContainsAgg(body)) body
                else rangeIR(entries.len).filter(!entries.at(_).isNA).streamAgg { i =>
                  M.agg {
                    M.sequence(
                      colName -> cols.at(i),
                      entryName -> entries.at(i),
                      body,
                    )
                  }
                }

            } yield newRow.insert(entriesFieldName -> entries)
          }
        }

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
            ToDict(mapIR(ToStream(uid)) { elt =>
              M.eval {
                for {
                  key <- elt.get("key")
                  value <- elt.get("value")
                  _ <- M.sequence(unwrapStruct(aggs, value).map(M.let[EVAL.type]): _*)
                } yield maketuple(key, liftedBody)
              }
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

        def cols = Ref(Name("__cols"), TArray(child.typ.colType))
        def colIdx = Ref(Name("__col_idx"), TInt32)

        val (
          setupOuterAggContext: M[EVAL.type],
          setupInnerAggContext: M[EVAL.type],
        ) =
          if (aggs.isEmpty) (M.unit, M.unit)
          else {
            val RunAgg =
              lc.deepCopy.aggregate { (global, row) =>
                M.eval {
                  for {
                    cols <- global.get(colsFieldName)
                    _ <- globalName -> global.select(child.typ.globalType.fieldNames)
                    result <- M.lift[AGG.type] {
                      for {
                        aggCols <- global.get(colsFieldName)
                        entries <- row.get(entriesFieldName)
                        _ <- globalName -> global.select(child.typ.globalType.fieldNames)
                        _ <- rowName -> row.select(child.typ.rowType.fieldNames)
                      } yield makestruct(
                        "__n_rows" ->
                          ApplyAggOp(Count())(),
                        "__array_aggs" ->
                          rangeIR(aggCols.len).toArray.aggElements(Some(cols.len)) { (elem, idx) =>
                            M.agg {
                              for {
                                _ <- colName -> aggCols.at(elem)
                                g <- entryName -> entries.at(elem)
                                _ <- M.lift[EVAL.type]((colName -> cols.at(idx)) >> M.unit)
                              } yield AggFilter(!g.isNA, bindingsToStruct(aggs), false)
                            }
                          },
                      )
                    }
                  } yield result
                }
              }

            val ident = freshName()
            liftedRelationalLets += (ident -> RunAgg)

            val arrayAggs = Ref(Name("__array_aggs"), RunAgg.get("__array_aggs").typ)

            val bindResult: M[EVAL.type] =
              for {
                result <- RelationalRef(ident, RunAgg.typ)
                _ <- arrayAggs.name -> result.get("__array_aggs")
                _ <- Name("n_rows") -> result.get("__n_rows")
                unit <- M.unit
              } yield unit

            val bindBody: M[EVAL.type] =
              arrayAggs.at(colIdx).flatMap { elem =>
                M.sequence(unwrapStruct(aggs, elem).map(M.let[EVAL.type]): _*)
              }

            (bindResult, bindBody)
          }

        val (
          setupOuterScanContext: M[EVAL.type],
          setupInnerScanContext: M[EVAL.type],
        ) =
          if (scans.isEmpty) (M.unit, M.unit)
          else {
            val RunScan = StreamAggScan(cols.stream, colName, bindingsToStruct(scans)).toArray
            val result = Ref(Name("__scan_result"), RunScan.typ)

            val bindResult: M[EVAL.type] =
              result.name -> RunScan

            val bindBody: M[EVAL.type] =
              result.at(colIdx).flatMap { elem =>
                M.sequence(unwrapStruct(scans, elem).map(M.let[EVAL.type]): _*)
              }

            (bindResult, bindBody)
          }

        lc.mapGlobals { global =>
          M.eval {
            for {
              global <- Name("__global_matrix_map_cols") -> global
              cols <- cols.name -> global.get(colsFieldName)
              _ <- globalName -> global.select(child.typ.globalType.fieldNames)
              _ <- setupOuterAggContext
              _ <- setupOuterScanContext
            } yield global.insert(
              colsFieldName -> ToArray(StreamMap(
                rangeIR(cols.len),
                colIdx.name,
                M.eval {
                  M.sequence(
                    colName -> cols.at(colIdx),
                    setupInnerAggContext,
                    setupInnerScanContext,
                    b0,
                  )
                },
              ))
            )
          }
        }

      case MatrixFilterEntries(child, pred) =>
        val mtype = child.typ
        lower(ctx, child, liftedRelationalLets).mapRows { (global, row) =>
          M.eval {
            for {
              cols <- global.get(colsFieldName)
              row <- Name("__row_matrix_filter_entries") -> row
              entries <- row.get(entriesFieldName)
              _ <- globalName -> global.select(mtype.globalType.fieldNames)
              _ <- rowName -> row.select(mtype.rowType.fieldNames)
            } yield row.insert(
              entriesFieldName ->
                ToArray(rangeIR(cols.len).streamMap { i =>
                  M.eval {
                    for {
                      _ <- colName -> cols.at(i)
                      g <- entryName -> entries.at(i)
                    } yield If(lower(ctx, pred, liftedRelationalLets), g, NA(mtype.entryType))
                  }
                })
            )
          }
        }

      case MatrixUnionCols(left, right, joinType) =>
        val ll = lower(ctx, left, liftedRelationalLets).distinct
        val rr = lower(ctx, right, liftedRelationalLets).distinct
        TableJoin(
          ll,
          rr
            .mapRows((_, row) => row.rename(_.rename(Map(entriesFieldName -> "__right_entries"))))
            .mapGlobals(global => makestruct("__right_cols" -> global.get(colsFieldName))),
          joinType,
        )
          .mapRows { (global, row) =>
            row
              .insert(
                entriesFieldName -> {
                  def handleMissingEntriesArray(entries: String, cols: String): IR =
                    if (joinType == "inner") row.get(entries)
                    else row.get(entries).orElse {
                      ToArray(rangeIR(global.get(cols).len).streamMap { _ =>
                        MakeStruct(right.typ.entryType.fields.map(f => f.name -> NA(f.typ)))
                      })
                    }

                  val ls = handleMissingEntriesArray(entriesFieldName, colsFieldName)
                  val rs = handleMissingEntriesArray("__right_entries", "__right_cols")
                  concatIR(ls, rs).toArray
                }
              )
              .drop("__right_entries")
          }
          .mapGlobals { global =>
            global
              .update(colsFieldName)(cols => concatIR(cols, global.get("__right_cols")).toArray)
              .drop("__right_cols")
          }

      case MatrixMapEntries(child, newEntries) =>
        lower(ctx, child, liftedRelationalLets).mapRows { (global, row) =>
          M.eval {
            for {
              cols <- global.get(colsFieldName)
              row <- Name("__row_matrix_map_entries") -> row
              entries <- row.get(entriesFieldName)
              _ <- globalName -> global.select(child.typ.globalType.fieldNames)
              _ <- rowName -> row.select(child.typ.rowType.fieldNames)
            } yield row.insert(
              entriesFieldName ->
                ToArray(StreamZip(
                  FastSeq(ToStream(cols), ToStream(entries)),
                  FastSeq(colName, entryName),
                  lower(ctx, newEntries, liftedRelationalLets),
                  ArrayZipBehavior.AssumeSameLength,
                ))
            )
          }
        }

      case MatrixRepartition(child, n, shuffle) =>
        TableRepartition(lower(ctx, child, liftedRelationalLets), n, shuffle)

      case MatrixFilterIntervals(child, intervals, keep) =>
        TableFilterIntervals(lower(ctx, child, liftedRelationalLets), intervals, keep)

      case MatrixUnionRows(children) =>
        // FIXME: this should check that all children have the same column keys.
        val first = lower(ctx, children.head, liftedRelationalLets)
        TableUnion(FastSeq(first) ++
          children.tail.map(lower(ctx, _, liftedRelationalLets).mapRows { (_, row) =>
            SelectFields(row, first.typ.rowType.fieldNames)
          }))

      case MatrixDistinctByRow(child) => TableDistinct(lower(ctx, child, liftedRelationalLets))

      case MatrixRowsHead(child, n) => TableHead(lower(ctx, child, liftedRelationalLets), n)
      case MatrixRowsTail(child, n) => TableTail(lower(ctx, child, liftedRelationalLets), n)

      case MatrixColsHead(child, n) =>
        lower(ctx, child, liftedRelationalLets)
          .mapGlobals(_.update(colsFieldName)(_.slice(0, Some(n))))
          .mapRows((_, row) => row.update(entriesFieldName)(_.slice(0, Some(n))))

      case MatrixColsTail(child, n) =>
        lower(ctx, child, liftedRelationalLets)
          .mapGlobals(_.update(colsFieldName)(_.slice(-n, None)))
          .mapRows((_, row) => row.update(entriesFieldName)(_.slice(-n, None)))

      case MatrixExplodeCols(child, path) =>
        lower(ctx, child, liftedRelationalLets)
          .mapGlobals { global =>
            global.get(colsFieldName).stream.streamAgg { col =>
              IRBuilder.scoped { ib =>
                val N = path.length

                val refs = new Array[Atom](N)
                val last = (0 until N).foldLeft(col) { (ref, i) =>
                  refs(i) = ref
                  ib.memoize(ref.get(path(i)), scope = AGG)
                }

                global.insert(
                  colsFieldName ->
                    last.stream.aggExplode { elt =>
                      ApplyAggOp(Collect())(
                        path.zip(refs.unsafeToArraySeq).foldRight[IR](elt) {
                          case ((p, ref), inserted) =>
                            ref.insert(p -> inserted)
                        }
                      )
                    },
                  "__lengths" ->
                    ApplyAggOp(Collect())(last.len),
                )
              }
            }
          }
          .mapRows { (global, row) =>
            M.eval {
              for {
                entries <- row.get(entriesFieldName)
                lengths <- global.get("__lengths")
              } yield row.insert(
                entriesFieldName ->
                  ToArray(rangeIR(entries.len).streamFlatMap { idx =>
                    rangeIR(lengths.at(idx)).streamMap(_ => entries.at(idx))
                  })
              )
            }
          }
          .mapGlobals(_.drop("__lengths"))

      case MatrixAggregateRowsByKey(child, entryExpr, rowExpr) =>
        lower(ctx, child, liftedRelationalLets).aggregateByKey { (global, row) =>
          M.eval {
            for {
              evalCols <- global.get(colsFieldName)
              _ <- globalName -> global.select(child.typ.globalType.fieldNames)
              newRow <- M.lift[AGG.type] {
                for {
                  aggCols <- global.get(colsFieldName)
                  entries <- row.get(entriesFieldName)
                  _ <- globalName -> global.select(child.typ.globalType.fieldNames)
                  _ <- rowName -> row.select(child.typ.rowType.fieldNames)
                } yield lower(ctx, rowExpr, liftedRelationalLets).insert(
                  entriesFieldName ->
                    rangeIR(aggCols.len).toArray.aggElements(Some(evalCols.len)) { (elem, idx) =>
                      M.agg {
                        for {
                          _ <- colName -> aggCols.at(elem)
                          g <- entryName -> entries.at(elem)
                          _ <- M.lift[EVAL.type]((colName -> evalCols.at(idx)) >> M.unit)
                          aggEntry = lower(ctx, entryExpr, liftedRelationalLets)
                        } yield AggFilter(!g.isNA, aggEntry, isScan = false)
                      }
                    }
                )
              }
            } yield newRow
          }
        }

      case MatrixCollectColsByKey(child) =>
        lower(ctx, child, liftedRelationalLets)
          .mapGlobals { global =>
            bindIR(global.get(colsFieldName)) { cols =>
              global.insert(
                "__new_col_idx" ->
                  rangeIR(cols.len)
                    .streamMap(i => maketuple(cols.at(i).select(child.typ.colKey), i))
                    .groupByKey
                    .stream
                    .toArray
              )
            }
          }
          .mapRows { (global, row) =>
            row.update(entriesFieldName) { entries =>
              mapArray(global.get("__new_col_idx")) { kv =>
                MakeStruct(child.typ.entryType.fieldNames.map { f =>
                  f -> mapArray(kv.get("value"))(i => entries.at(i).get(f))
                })
              }
            }
          }
          .mapGlobals { global =>
            global.update(colsFieldName) { cols =>
              mapArray(global.get("__new_col_idx")) { kv =>
                InsertFields(
                  kv.get("key"),
                  child.typ.colValueStruct.fieldNames.map { f =>
                    f -> mapArray(kv.get("value"))(i => cols.at(i).get(f))
                  },
                )
              }
            }
              .drop("__new_col_idx")
          }

      case MatrixExplodeRows(child, path) =>
        TableExplode(lower(ctx, child, liftedRelationalLets), path)

      case mr: MatrixRead => mr.lower(ctx)

      case MatrixAggregateColsByKey(child, entryExpr, colExpr) =>
        lower(ctx, child, liftedRelationalLets)
          .mapGlobals { global =>
            bindIR(global.get(colsFieldName)) { cols =>
              global.insert(
                "__key_map" ->
                  rangeIR(cols.len)
                    .streamMap(idx => maketuple(cols.at(idx).select(child.typ.colKey), idx))
                    .groupByKey
                    .stream
                    .toArray
              )
            }
          }
          .mapRows { (global, row) =>
            M.eval {
              for {
                keyMap <- global.get("__key_map")
                cols <- global.get(colsFieldName)
                row <- Name("__row_matrix_aggregate_cols_by_key") -> row
                entries <- row.get(entriesFieldName)
                _ <- globalName -> global.select(child.typ.globalType.fieldNames)
                _ <- rowName -> row.select(child.typ.rowType.fieldNames)
                newEntries <- ToArray(rangeIR(keyMap.len).streamMap { idx =>
                  keyMap.at(idx).get("value").stream.streamAgg { aggIdx =>
                    M.agg {
                      for {
                        _ <- colName -> cols.at(aggIdx)
                        g <- entryName -> entries.at(aggIdx)
                        aggEntries = lower(ctx, entryExpr, liftedRelationalLets)
                      } yield AggFilter(!g.isNA, aggEntries, isScan = false)
                    }
                  }
                })
              } yield row.insert(entriesFieldName -> newEntries)
            }
          }
          .mapGlobals { global =>
            M.eval {
              for {
                cols <- global.get(colsFieldName)
                keyMap <- global.get("__key_map")
                global <- Name("__global_matrix_aggregate_cols_by_key") -> global
                _ <- globalName -> global.select(child.typ.globalType.fieldNames)
                newCols <- ToArray(rangeIR(keyMap.len)
                  .streamMap { idx =>
                    M.eval {
                      for {
                        elem <- keyMap.at(idx)
                        key <- elem.get("key")
                        value <- elem.get("value").stream.streamAgg { aggIdx =>
                          M.agg {
                            (colName -> cols.at(aggIdx)) >>
                              lower(ctx, colExpr, liftedRelationalLets)
                          }
                        }
                      } yield key.insert(
                        value.typ.asInstanceOf[TStruct].fieldNames.map(f => f -> value.get(f))
                      )
                    }
                  })
              } yield global.insert(colsFieldName -> newCols).drop("__key_map")
            }
          }

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
          .mapRows((_, row) => row.select(child.typ.rowType.fieldNames :+ entriesFieldName))
          .mapGlobals(_.select(child.typ.globalType.fieldNames :+ colsFieldName))
          .rename(Map(entriesFieldName -> entries), Map(colsFieldName -> cols))

      case x @ MatrixEntriesTable(child) =>
        val lc = lower(ctx, child, ab)

        if (child.typ.rowKey.nonEmpty && child.typ.colKey.nonEmpty) {
          lc
            .mapGlobals { global =>
              bindIR(global.get(colsFieldName)) { cols =>
                global.insert(
                  "__old_col_idx" ->
                    rangeIR(cols.len)
                      .streamMap(idx => maketuple(cols.at(idx).select(child.typ.colKey), idx))
                      .sort(ascending = true, onKey = true)
                      .stream
                      .streamMap(_.get(1))
                      .toArray
                )
              }
            }
            .aggregateByKey { (_, row) =>
              makestruct(
                "__values" ->
                  ApplyAggOp(Collect())(row.select(lc.typ.valueType.fieldNames))
              )
            }
            .mapRows { (global, row) =>
              bindIR(global.get(colsFieldName)) { cols =>
                row.drop("__values").insert(
                  "__explode" ->
                    ToArray(global.get("__old_col_idx").stream.streamFlatMap { oldColIndex =>
                      bindIR(cols.at(oldColIndex)) { col =>
                        row.get("__values").stream.streamFlatMap { v =>
                          bindIR(v.get(entriesFieldName).at(oldColIndex)) { entry =>
                            val newRow =
                              MakeStruct(
                                child.typ.rowValueStruct.fieldNames.map(f => f -> v.get(f)) ++
                                  child.typ.colType.fieldNames.map(f => f -> col.get(f)) ++
                                  child.typ.entryType.fieldNames.map(f => f -> entry.get(f))
                              )

                            If(IsNA(entry), MakeStream.empty(newRow.typ), MakeStream(newRow))
                          }
                        }
                      }
                    })
                )
              }
            }
            .explode("__explode")
            .mapRows { (_, row) =>
              bindIR(row.get("__explode")) { exploded =>
                MakeStruct(x.typ.rowType.fieldNames.map { f =>
                  f -> (if (child.typ.rowKey.contains(f)) row.get(f) else exploded.get(f))
                })
              }
            }
            .mapGlobals(_.drop(colsFieldName, "__old_col_idx"))
            .keyBy(child.typ.rowKey ++ child.typ.colKey, isSorted = true)
        } else {
          val result =
            lc
              .mapRows { (global, row) =>
                bindIR(row.get(entriesFieldName)) { entries =>
                  row.insert(
                    "__col_idx" ->
                      ToArray(rangeIR(global.get(colsFieldName).len).filter(!entries.at(_).isNA))
                  )
                }
              }
              .explode("__col_idx")
              .mapRows { (global, row) =>
                M.eval {
                  for {
                    colIdx <- row.get("__col_idx")
                    colStruct <- global.get(colsFieldName).at(colIdx)
                    entryStruct <- row.get(entriesFieldName).at(colIdx)

                    newFields =
                      child.typ.colType.fieldNames.map(f => f -> colStruct.get(f)) ++
                        child.typ.entryType.fieldNames.map(f => f -> entryStruct.get(f))

                  } yield row
                    .drop(entriesFieldName, "__col_idx")
                    .insert(newFields, ordering = Some(x.typ.rowType.fieldNames))
                }
              }
              .mapGlobals(_.drop(colsFieldName))

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
          .mapGlobals(_.drop(colsFieldName))
          .mapRows((_, row) => row.drop(entriesFieldName))

      case MatrixColsTable(child) =>
        val colKey = child.typ.colKey

        bindIR(lower(ctx, child, ab).getGlobals) { global =>
          val sortedCols =
            if (colKey.isEmpty) global.get(colsFieldName)
            else global
              .get(colsFieldName)
              .stream
              .streamMap(elem => maketuple(elem.select(colKey), elem))
              .sort(ascending = true, onKey = true)
              .stream
              .streamMap(_.get(1))
              .toArray

          makestruct("rows" -> sortedCols, "global" -> global.drop(colsFieldName))
        }
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
        lower(ctx, child, ab).aggregate { (global, _) =>
          maketuple(ApplyAggOp(Count())(), global.get(colsFieldName).len)
        }
      case MatrixAggregate(child, query) =>
        lower(ctx, child, ab).aggregate { (global, row) =>
          M.agg {
            for {
              cols <- global.get(colsFieldName)
              entries <- row.get(entriesFieldName)
              _ <- globalName -> global.select(child.typ.globalType.fieldNames)
              _ <- rowName -> row.select(child.typ.rowType.fieldNames)
              _ <- M.lift[EVAL.type] {
                (globalName -> global.select(child.typ.globalType.fieldNames)) >>
                  M.unit
              }
            } yield zip2(
              ToStream(cols),
              ToStream(entries),
              ArrayZipBehavior.AssertSameLength,
            ) {
              (c, e) => maybeIR(e)(e => maketuple(c, e))
            }
              .filter(!_.isNA)
              .aggExplode { explodedTuple =>
                M.agg {
                  M.sequence(
                    colName -> GetTupleElement(explodedTuple, 0),
                    entryName -> GetTupleElement(explodedTuple, 1),
                    query,
                  )
                }
              }
          }
        }
      case _ =>
        lowerChildren(ctx, ir, ab).asInstanceOf[IR]
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

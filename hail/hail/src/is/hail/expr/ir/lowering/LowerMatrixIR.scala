package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.collection.implicits.toRichArray
import is.hail.expr.ir.{Memoized => M, _}
import is.hail.expr.ir.MatrixIR.{colName, entryName, globalName, rowName}
import is.hail.expr.ir.Scope._
import is.hail.expr.ir.Scopes.ScopesSyntax
import is.hail.expr.ir.defs._
import is.hail.expr.ir.defs.ArrayZipBehavior._
import is.hail.expr.ir.functions.{WrappedMatrixToTableFunction, WrappedMatrixToValueFunction}
import is.hail.types.virtual._
import is.hail.utils._

import scala.collection.immutable.ArraySeq
import scala.collection.mutable

object LowerMatrixIR extends Logging {
  val entriesFieldName: String = MatrixType.entriesIdentifier
  val colsFieldName: String = "__cols"

  def apply(ctx: ExecuteContext, ir0: BaseIR): BaseIR = {
    val lowered =
      ir0 match {
        case ir: IR => lower(ctx, ir)
        case tir: TableIR => lower(ctx, tir)
        case mir: MatrixIR => lower(ctx, mir)
        case bmir: BlockMatrixIR => lower(ctx, bmir)
      }

    if (lowered ne ir0) NormalizeNames()(ctx, lowered)
    else ir0
  }

  private def lowerChildren(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    ir.mapChildren {
      case tir: TableIR => lower(ctx, tir)
      case mir: MatrixIR => throw new RuntimeException(s"expect specialized lowering rule for " +
          s"${ir.getClass.getName}\n  Found MatrixIR child $mir")
      case bmir: BlockMatrixIR => lower(ctx, bmir)
      case vir: IR => lower(ctx, vir)
    }

  // newRow and newCol can hold both aggs and scans. The table node each one lowers to
  // supports only one of the two. A stream node does the other.
  //
  //   We lower MatrixMapRows to TableMapRows. That supports scans over rows but not aggs.
  //     Aggs: run in a StreamAgg over the row's entries, wrapped around the body.
  //     Scans: a StreamAgg has no scan scope, so they are lifted out of it, to the top of
  //       the TableMapRows body.
  //
  //   We lower MatrixMapCols to TableMapGlobals. That supports aggs over rows but not scans.
  //     Aggs: run there, once per col, in an AggArrayPerElement over the cols.
  //     Scans: lifted into a StreamAggScan over the cols array.
  //
  // To lift a scan is to move it to where it can run, bind its result to a fresh name there,
  // and leave a Ref to that name where the scan was.
  //
  // A scan's init args must be invariant on rows or cols. They are evaluated once, before
  // the scan sees its first row or col. `hoist` enforces this when moving an init arg and
  // its dependencies. This is done here, rather than left to Extract, because a lifted scan
  // has left the binders it read from, and Extract can no longer see what was wrong.
  //
  // Lifting uses an environment, Name -> LiftState, that tracks:
  //   Invariant: bound outside the body, like `global`; an init arg can read it.
  //   Movable(v, env): an EVAL binding, bound in env; invariant iff its lifted value v is.
  //   Varying: varies per row, column, loop iteration or element; an init arg can't read it.
  sealed private trait LiftState
  private case object Invariant extends LiftState
  final private case class Movable(v: IR, env: LiftEnv) extends LiftState
  private case object Varying extends LiftState

  private type LiftEnv = BindingEnv[LiftState]

  private def bindingsToStruct(bindings: IndexedSeq[(Name, IR)]): MakeStruct =
    MakeStruct(bindings.map { case (n, ir) => n.str -> ir })

  private def unwrapM(bindings: IndexedSeq[(Name, _)], struct: Atom): M =
    bindings.foldLeft(M.unit) { case (m, (name, _)) => m >> (name -> struct.get(name.str)) }

  private def liftScans(ir: IR, env: LiftEnv, onCols: Boolean): (IR, M, IndexedSeq[(Name, IR)]) = {
    val (node, what) = if (onCols) ("MatrixMapCols", "newCol") else ("MatrixMapRows", "newRow")
    var top: M = M.unit
    // Names of bindings `hoist` has moved to `top`. Each is moved once.
    val moved = mutable.Set.empty[Name]
    // Names of lifted scans' results. They aren't in the env, so `hoist` checks them here.
    val lifted = mutable.Set.empty[Name]

    def extendEnv(env: LiftEnv, ir: IR, i: Int): LiftEnv =
      ChildEnv.get(env, ir, i)((_, _) => Varying)

    def noAggs(x: IR): Unit =
      if (ContainsAgg(x)) fatal(s"aggregations inside scans are not supported in $node")

    // `env` must be the env of the place `x` sits, so that it binds every name `x` reads.
    def hoist(x: IR, env: LiftEnv): Unit = {
      noAggs(x)
      FreeVariables(x, supportsAgg = false, supportsScan = true).eval.m.keys.foreach { n =>
        if (lifted.contains(n)) fatal(s"scan init argument depends on a scan result: $n")
        env.eval.lookup(n) match {
          case Invariant =>
          case Movable(v, env) => if (moved.add(n)) { hoist(v, env); top = top >> M.let(n -> v) }
          case Varying => fatal(s"scan init argument refers to $n, which can't leave $what")
        }
      }
    }

    def record(builder: mutable.Growable[(Name, IR)], name: Name, x: IR): Unit = {
      lifted += name
      builder += (name -> x)
    }

    // A scan combinator. The scans inside it were hoisted where they sat.
    def emitCombinator(builder: mutable.Growable[(Name, IR)], name: Name, x: IR): Unit = {
      noAggs(x)
      record(builder, name, x)
    }

    // An agg's init args are evaluated once, but on cols a scan's result differs per col.
    def scanInAggInit(ir: IR): Boolean =
      onCols && (ir match {
        case a: ApplyAggOp => a.initOpArgs.exists(ContainsScan(_))
        case AggFold(zero, _, _, _, _, false) => ContainsScan(zero)
        case AggArrayPerElement(_, _, _, _, knownLength, false) =>
          knownLength.exists(ContainsScan(_))
        case _ => false
      })

    // Lifts the scans in a combinator's `body`, then records the combinator as one scan.
    // `wrap` rebuilds the combinator around the struct of the scans found in `body`.
    // Returns the lifted body, the scans found in it, and a Ref to the combinator's result.
    def liftCombinator(
      body: IR,
      bodyEnv: LiftEnv,
      builder: mutable.Growable[(Name, IR)],
    )(
      wrap: IR => IR
    ): (IR, IndexedSeq[(Name, IR)], Ref) = {
      val ab = ArraySeq.newBuilder[(Name, IR)]
      val liftedBody = lift(body, bodyEnv, ab)
      val inner = ab.result()
      val x = wrap(bindingsToStruct(inner))
      val uid = Ref(freshName(), x.typ)
      emitCombinator(builder, uid.name, x)
      (liftedBody, inner, uid)
    }

    def lift(ir: IR, env: LiftEnv, builder: mutable.Growable[(Name, IR)]): IR =
      ir match {
        case _ if scanInAggInit(ir) =>
          fatal(s"scans inside aggregations are not supported in $node")

        // A scan with init args of its own.
        case _: ApplyScanOp | AggFold(_, _, _, _, _, true) =>
          hoist(ir, env)
          val s = freshName()
          record(builder, s, ir)
          Ref(s, ir.typ)

        case AggFilter(filt, body, true) =>
          val (liftedBody, inner, uid) =
            liftCombinator(body, extendEnv(env, ir, 1), builder)(AggFilter(filt, _, true))
          M.eval(unwrapM(inner, uid) >> liftedBody)

        case AggExplode(a, name, body, true) =>
          val (liftedBody, inner, uid) =
            liftCombinator(body, extendEnv(env, ir, 1), builder)(AggExplode(a, name, _, true))
          M.eval(unwrapM(inner, uid) >> liftedBody)

        case AggGroupBy(a, body, true) =>
          val (liftedBody, inner, uid) =
            liftCombinator(body, extendEnv(env, ir, 1), builder)(AggGroupBy(a, _, true))

          ToDict(mapIR(ToStream(uid)) { elt =>
            M.eval {
              for {
                key <- elt.get("key")
                value <- elt.get("value")
                _ <- unwrapM(inner, value)
              } yield maketuple(key, liftedBody)
            }
          })

        case AggArrayPerElement(a, elementName, indexName, body, knownLength, true) =>
          val (liftedBody, inner, uid) =
            liftCombinator(body, extendEnv(env, ir, 1), builder)(
              AggArrayPerElement(a, elementName, indexName, _, knownLength, true)
            )

          // `knownLength` is this combinator's one init arg.
          knownLength.foreach(hoist(_, env))

          mapArray(uid)(eltUID => M.eval(unwrapM(inner, eltUID) >> liftedBody))

        case Block(bindings, body) =>
          def go(i: Int, env: LiftEnv, builder: mutable.Growable[(Name, IR)]): M =
            if (i == bindings.length) lift(body, env, builder)
            else bindings(i) match {
              case b @ Binding(name, _, SCAN) =>
                val ab = ArraySeq.newBuilder[(Name, IR)]
                val liftedBody = go(i + 1, env.bindScan(name, Varying), ab)
                val aggs = ab.result()
                val structResult = bindingsToStruct(aggs)
                val uid = Ref(freshName(), structResult.typ)
                emitCombinator(builder, uid.name, Block(FastSeq(b), structResult))
                unwrapM(aggs, uid) >> liftedBody
              case Binding(name, value, scope) =>
                val v = lift(value, Binds.aggSeq(env, scope), builder)
                val state = if (scope == EVAL) Movable(v, env) else Varying
                val e = env.bind(scope, FastSeq(name -> state))
                val liftedBody = go(i + 1, e, builder)
                if (moved.contains(name)) liftedBody
                else M.interleave(scope)(name -> v) >> liftedBody
            }

          M.eval(go(0, env, builder))

        case StreamAggScan(a, name, query) =>
          StreamAggScan(lift(a, extendEnv(env, ir, 0), builder), name, query)

        case _: TableAggregate | _: MatrixAggregate =>
          ir

        case _ =>
          ir.mapChildrenWithEnv(env)((_, _) => Varying) {
            case (c: IR, e) => lift(c, e, builder)
            case (c, _) => c
          }
      }

    val builder = ArraySeq.newBuilder[(Name, IR)]
    val body = lift(ir, env, builder)
    (body, top, builder.result())
  }

  private def lower(ctx: ExecuteContext, mir: MatrixIR): TableIR = {
    val lowered = mir match {
      case CastTableToMatrix(child, entries, cols, _) =>
        lower(ctx, child)
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
        val loweredChild = lower(ctx, child)
        TableToTableApply(loweredChild, function.lower())

      case MatrixRename(child, globalMap, colMap, rowMap, entryMap) =>
        var t = lower(ctx, child).rename(rowMap, globalMap)

        if (colMap.nonEmpty)
          t = t.mapGlobals((global, _) =>
            global.rename(_.insertFields(FastSeq(
              colsFieldName -> TArray(child.typ.colType.rename(colMap))
            )))
          )

        if (entryMap.nonEmpty)
          t = t.mapRows { (_, row) =>
            row.rename(_.insertFields(FastSeq(
              entriesFieldName -> TArray(child.typ.entryType.rename(entryMap))
            )))
          }

        t

      case MatrixKeyRowsBy(child, keys, isSorted) =>
        lower(ctx, child).keyBy(keys, isSorted)

      case MatrixFilterRows(child, pred) =>
        lower(ctx, child).filter { (global, row) =>
          M.eval {
            M.sequence(
              globalName -> global.select(child.typ.globalType.fieldNames),
              rowName -> row.select(child.typ.rowType.fieldNames),
              lower(ctx, pred),
            )
          }
        }

      case MatrixFilterCols(child, pred) =>
        lower(ctx, child)
          .mapGlobals((global, _) =>
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
                      lower(ctx, pred),
                    )
                  })
              )
            }
          )
          .mapRows { (global, row) =>
            row.update(entriesFieldName) { entries =>
              mapArray(global.get("__new_col_idx"))(entries.at(_))
            }
          }
          .mapGlobals((global, _) =>
            global
              .update(colsFieldName)(cols => mapArray(global.get("__new_col_idx"))(cols.at(_)))
              .drop("__new_col_idx")
          )

      case MatrixAnnotateRowsTable(child, table, root, product) =>
        val kt = table.typ.keyType
        if (kt.size == 1 && kt.types(0) == TInterval(child.typ.rowKeyStruct.types(0)))
          TableIntervalJoin(
            lower(ctx, child),
            lower(ctx, table),
            root,
            product,
          )
        else
          TableLeftJoinRightDistinct(
            lower(ctx, child),
            lower(ctx, table),
            root,
          )

      case MatrixChooseCols(child, oldIndices) =>
        lower(ctx, child)
          .mapGlobals((g, _) => g.insert("__new_col_idx" -> Literal(TArray(TInt32), oldIndices)))
          .mapRows { (global, row) =>
            row.update(entriesFieldName) { entries =>
              mapArray(global.get("__new_col_idx"))(entries.at(_))
            }
          }
          .mapGlobals((global, _) =>
            global
              .update(colsFieldName)(cols => mapArray(global.get("__new_col_idx"))(cols.at(_)))
              .drop("__new_col_idx")
          )

      case MatrixAnnotateColsTable(child, table, root) =>
        lower(ctx, child).mapGlobals((global, _) =>
          lower(ctx, table).collectAsDict.bind { annotations =>
            global.update(colsFieldName) { cols =>
              mapArray(cols) { col =>
                val key =
                  col.select(child.typ.colKey).rename(
                    _.rename(child.typ.colKey.zip(table.typ.key).toMap)
                  )

                col.insert(root -> annotations.invoke("get", table.typ.valueType, key))
              }
            }
          }
        )

      case MatrixMapGlobals(child, newGlobals) =>
        lower(ctx, child).mapGlobals((global, _) =>
          M.eval {
            for {
              cols <- global.get(colsFieldName)
              _ <- globalName -> global.select(child.typ.globalType.fieldNames)
              newGlobals <- lower(ctx, newGlobals)
            } yield newGlobals.insert(colsFieldName -> cols)
          }
        )

      case MatrixMapRows(child, newRow) =>
        val lc = lower(ctx, child)

        val (body, scanTop, scans) =
          liftScans(
            lower(ctx, newRow),
            Binds.matrixMapRows(
              BindingEnv.empty[LiftState],
              global = Invariant,
              sa = Varying,
              va = Varying,
              g = Varying,
              nCols = Invariant,
            ),
            onCols = false,
          )

        lc.mapRows { (global, row) =>
          val bindScans: M =
            M.sequence(
              M.interleave(SCAN)(globalName -> global.select(child.typ.globalType.fieldNames)),
              M.interleave(SCAN)(rowName -> row.select(child.typ.rowType.fieldNames)),
              scans.foldLeft(scanTop)(_ >> M.let(_)),
            )

          def aggregateEntries(cols: Atom, entries: Atom): IR =
            cols
              .stream
              .zip(entries.stream)((c, e) => maybeIR(e)(maketuple(c, _)))
              .filter(!_.isNA)
              .streamAgg { elem =>
                M.agg(M.sequence(colName -> elem.get(0), entryName -> elem.get(1), body))
              }

          M.eval {
            for {
              entries <- row.get(entriesFieldName)
              cols <- global.get(colsFieldName)
              _ <- Name("n_cols") -> cols.len
              _ <- globalName -> global.select(child.typ.globalType.fieldNames)
              _ <- rowName -> row.select(child.typ.rowType.fieldNames)
              _ <- M.when(scans.nonEmpty)(bindScans)
              newRow <- if (ContainsAgg(body)) aggregateEntries(cols, entries) else body
            } yield newRow.insert(entriesFieldName -> entries)
          }
        }

      case MatrixMapCols(child, newCol, _) =>
        val lc = lower(ctx, child)

        // `n_rows` is an aggregation result, so no scan init arg may read it.
        val (body, scanTop, scans) =
          liftScans(
            lower(ctx, newCol),
            Binds.matrixMapCols(
              BindingEnv.empty[LiftState],
              global = Invariant,
              sa = Varying,
              va = Varying,
              g = Varying,
              nRows = Varying,
            ),
            onCols = true,
          )

        val hasAggs = ContainsAgg(body)

        lc.mapGlobals { (global, row) =>
          val scanResult = Ref(Name("__scan_result"), TArray(bindingsToStruct(scans).typ))

          def aggregateCol(cols: Atom, entries: Atom)(col: Atom, idx: Atom): IR =
            M.agg {
              for {
                _ <- colName -> col
                g <- entryName -> entries.at(idx)
                _ <- M.interleave(EVAL)(
                  (colName -> cols.at(idx)) >>
                    M.when(scans.nonEmpty)(scanResult.at(idx).flatMap(unwrapM(scans, _)))
                )
              } yield AggFilter(!g.isNA, body, isScan = false)
            }

          def aggregateCols(cols: Atom): M =
            M.interleave(AGG) {
              for {
                aggCols <- global.get(colsFieldName)
                entries <- row.get(entriesFieldName)
                _ <- globalName -> global.select(child.typ.globalType.fieldNames)
                _ <- rowName -> row.select(child.typ.rowType.fieldNames)
              } yield aggCols.aggElements(Some(cols.len))(aggregateCol(cols, entries))
            }

          def mapCols(cols: Atom): M =
            if (!hasAggs)
              if (scans.isEmpty) StreamMap(cols.stream, colName, body).toArray
              else StreamAggScan(cols.stream, colName, Let(scans, body)).toArray
            else M.sequence(
              Name("n_rows") -> ApplyAggOp(Count())(), // bound only when newCol has aggs
              M.when(scans.nonEmpty)(
                scanResult.name ->
                  StreamAggScan(cols.stream, colName, bindingsToStruct(scans)).toArray
              ),
              aggregateCols(cols),
            )

          M.eval {
            for {
              // `global` stays the table's globals: `mapCols` reads it in the agg scope
              oldGlobal <- Name("__global_matrix_map_cols") -> global
              cols <- oldGlobal.get(colsFieldName)
              _ <- globalName -> oldGlobal.select(child.typ.globalType.fieldNames)
              _ <- scanTop
              newCols <- mapCols(cols)
            } yield oldGlobal.insert(colsFieldName -> newCols)
          }
        }

      case MatrixFilterEntries(child, pred) =>
        val mtype = child.typ
        lower(ctx, child).mapRows { (global, row) =>
          M.eval {
            for {
              cols <- global.get(colsFieldName)
              row <- Name("__row_matrix_filter_entries") -> row
              entries <- row.get(entriesFieldName)
              _ <- globalName -> global.select(mtype.globalType.fieldNames)
              _ <- rowName -> row.select(mtype.rowType.fieldNames)
            } yield row.insert(
              entriesFieldName ->
                ToArray(cols.stream.zip(entries.stream) { (col, entry) =>
                  M.eval {
                    for {
                      _ <- colName -> col
                      g <- entryName -> entry
                    } yield If(lower(ctx, pred), g, NA(mtype.entryType))
                  }
                })
            )
          }
        }

      case MatrixUnionCols(left, right, joinType) =>
        val ll = lower(ctx, left).distinct
        val rr = lower(ctx, right).distinct
        TableJoin(
          ll,
          rr
            .mapRows((_, row) => row.rename(_.rename(Map(entriesFieldName -> "__right_entries"))))
            .mapGlobals((global, _) => makestruct("__right_cols" -> global.get(colsFieldName))),
          joinType,
        )
          .mapRows { (global, row) =>
            row
              .insert(
                entriesFieldName -> {
                  def handleMissingEntriesArray(entries: String, cols: String): IR =
                    if (joinType == "inner") row.get(entries)
                    else row.get(entries).orElse {
                      ToArray(global.get(cols).stream.streamMap { _ =>
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
          .mapGlobals((global, _) =>
            global
              .update(colsFieldName)(cols => concatIR(cols, global.get("__right_cols")).toArray)
              .drop("__right_cols")
          )

      case MatrixMapEntries(child, newEntries) =>
        lower(ctx, child).mapRows { (global, row) =>
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
                  lower(ctx, newEntries),
                  AssumeSameLength,
                ))
            )
          }
        }

      case MatrixRepartition(child, n, shuffle) =>
        TableRepartition(lower(ctx, child), n, shuffle)

      case MatrixFilterIntervals(child, intervals, keep) =>
        TableFilterIntervals(lower(ctx, child), intervals, keep)

      case MatrixUnionRows(children) =>
        // FIXME: this should check that all children have the same column keys.
        val first = lower(ctx, children.head)
        TableUnion(FastSeq(first) ++
          children.tail.map(lower(ctx, _).mapRows { (_, row) =>
            SelectFields(row, first.typ.rowType.fieldNames)
          }))

      case MatrixDistinctByRow(child) => TableDistinct(lower(ctx, child))

      case MatrixRowsHead(child, n) => TableHead(lower(ctx, child), n)
      case MatrixRowsTail(child, n) => TableTail(lower(ctx, child), n)

      case MatrixColsHead(child, n) =>
        lower(ctx, child)
          .mapGlobals((g, _) => g.update(colsFieldName)(_.slice(0, Some(n))))
          .mapRows((_, row) => row.update(entriesFieldName)(_.slice(0, Some(n))))

      case MatrixColsTail(child, n) =>
        lower(ctx, child)
          .mapGlobals((g, _) => g.update(colsFieldName)(_.slice(-n, None)))
          .mapRows((_, row) => row.update(entriesFieldName)(_.slice(-n, None)))

      case MatrixExplodeCols(child, path) =>
        lower(ctx, child)
          .mapGlobals((global, _) =>
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
          )
          .mapRows { (global, row) =>
            M.eval {
              for {
                entries <- row.get(entriesFieldName)
                lengths <- global.get("__lengths")
              } yield row.insert(
                entriesFieldName ->
                  entries
                    .stream
                    .zip(lengths.stream)((entry, n) => rangeIR(n).streamMap(_ => entry))
                    .streamFlatten
                    .toArray
              )
            }
          }
          .mapGlobals((g, _) => g.drop("__lengths"))

      case MatrixAggregateRowsByKey(child, entryExpr, rowExpr) =>
        lower(ctx, child).aggregateByKey { (global, row) =>
          M.eval {
            for {
              evalCols <- global.get(colsFieldName)
              _ <- globalName -> global.select(child.typ.globalType.fieldNames)
              newRow <- M.interleave(AGG) {
                for {
                  aggCols <- global.get(colsFieldName)
                  entries <- row.get(entriesFieldName)
                  _ <- globalName -> global.select(child.typ.globalType.fieldNames)
                  _ <- rowName -> row.select(child.typ.rowType.fieldNames)
                } yield lower(ctx, rowExpr).insert(
                  entriesFieldName ->
                    rangeIR(aggCols.len).toArray.aggElements(Some(evalCols.len)) { (elem, idx) =>
                      M.agg {
                        for {
                          _ <- colName -> aggCols.at(elem)
                          g <- entryName -> entries.at(elem)
                          _ <- M.interleave(EVAL)(colName -> evalCols.at(idx))
                          aggEntry = lower(ctx, entryExpr)
                        } yield AggFilter(!g.isNA, aggEntry, isScan = false)
                      }
                    }
                )
              }
            } yield newRow
          }
        }

      case MatrixCollectColsByKey(child) =>
        lower(ctx, child)
          .mapGlobals((global, _) =>
            bindIR(global.get(colsFieldName)) { cols =>
              global.insert(
                "__new_col_idx" ->
                  cols
                    .stream
                    .streamMap(_.select(child.typ.colKey))
                    .zipWithIndex
                    .groupByKey
                    .stream
                    .toArray
              )
            }
          )
          .mapRows { (global, row) =>
            row.update(entriesFieldName) { entries =>
              mapArray(global.get("__new_col_idx")) { kv =>
                MakeStruct(child.typ.entryType.fieldNames.map { f =>
                  f -> mapArray(kv.get("value"))(entries.at(_).get(f))
                })
              }
            }
          }
          .mapGlobals((global, _) =>
            global.update(colsFieldName) { cols =>
              mapArray(global.get("__new_col_idx")) { kv =>
                InsertFields(
                  kv.get("key"),
                  child.typ.colValueStruct.fieldNames.map { f =>
                    f -> mapArray(kv.get("value"))(cols.at(_).get(f))
                  },
                )
              }
            }
              .drop("__new_col_idx")
          )

      case MatrixExplodeRows(child, path) =>
        TableExplode(lower(ctx, child), path)

      case mr: MatrixRead => mr.lower(ctx)

      case MatrixAggregateColsByKey(child, entryExpr, colExpr) =>
        lower(ctx, child)
          .mapGlobals((global, _) =>
            bindIR(global.get(colsFieldName)) { cols =>
              global.insert(
                "__key_map" ->
                  cols
                    .stream
                    .streamMap(_.select(child.typ.colKey))
                    .zipWithIndex
                    .groupByKey
                    .stream
                    .toArray
              )
            }
          )
          .mapRows { (global, row) =>
            M.eval {
              for {
                keyMap <- global.get("__key_map")
                cols <- global.get(colsFieldName)
                row <- Name("__row_matrix_aggregate_cols_by_key") -> row
                entries <- row.get(entriesFieldName)
                _ <- globalName -> global.select(child.typ.globalType.fieldNames)
                _ <- rowName -> row.select(child.typ.rowType.fieldNames)
                newEntries <- mapArray(keyMap) { kv =>
                  kv.get("value").stream.streamAgg { aggIdx =>
                    M.agg {
                      for {
                        _ <- colName -> cols.at(aggIdx)
                        g <- entryName -> entries.at(aggIdx)
                        aggEntries = lower(ctx, entryExpr)
                      } yield AggFilter(!g.isNA, aggEntries, isScan = false)
                    }
                  }
                }
              } yield row.insert(entriesFieldName -> newEntries)
            }
          }
          .mapGlobals((global, _) =>
            M.eval {
              for {
                cols <- global.get(colsFieldName)
                keyMap <- global.get("__key_map")
                global <- Name("__global_matrix_aggregate_cols_by_key") -> global
                _ <- globalName -> global.select(child.typ.globalType.fieldNames)
                newCols <- ToArray(keyMap.stream.streamMap { elem =>
                  val aggValue =
                    elem.get("value").stream.streamAgg { aggIdx =>
                      M.agg {
                        (colName -> cols.at(aggIdx)) >> lower(ctx, colExpr)
                      }
                    }

                  aggValue.bind { value =>
                    elem.get("key").insert(
                      value.typ.asInstanceOf[TStruct].fieldNames.map(f => f -> value.get(f))
                    )
                  }
                })
              } yield global.insert(colsFieldName -> newCols).drop("__key_map")
            }
          )

      case MatrixLiteral(_, tl) => tl
    }

    if (!mir.typ.isCompatibleWith(lowered.typ))
      throw new RuntimeException(
        s"Lowering changed type:" +
          s"\n  BEFORE: ${Pretty(ctx, mir).strip}" +
          s"\n    ${mir.typ}" +
          s"\n    ${mir.typ.canonicalTableType}" +
          s"\n  AFTER: ${Pretty(ctx, lowered).strip}" +
          s"\n    ${lowered.typ}"
      )

    lowered
  }

  private def lower(ctx: ExecuteContext, tir: TableIR): TableIR = {
    val lowered = tir match {
      case CastMatrixToTable(child, entries, cols) =>
        lower(ctx, child)
          .mapRows((_, row) => row.select(child.typ.rowType.fieldNames :+ entriesFieldName))
          .mapGlobals((g, _) => g.select(child.typ.globalType.fieldNames :+ colsFieldName))
          .rename(Map(entriesFieldName -> entries), Map(colsFieldName -> cols))

      case x @ MatrixEntriesTable(child) =>
        val lc = lower(ctx, child)

        if (child.typ.rowKey.nonEmpty && child.typ.colKey.nonEmpty) {
          lc
            .mapGlobals((global, _) =>
              bindIR(global.get(colsFieldName)) { cols =>
                global.insert(
                  "__old_col_idx" ->
                    cols
                      .stream
                      .streamMap(_.select(child.typ.colKey))
                      .zipWithIndex
                      .sort(ascending = true, onKey = true)
                      .stream
                      .streamMap(_.get("idx"))
                      .toArray
                )
              }
            )
            .aggregateByKey { (_, row) =>
              makestruct(
                "__values" ->
                  ApplyAggOp(Collect())(row.select(lc.typ.valueType.fieldNames))
              )
            }
            .mapRows { (global, row) =>
              bindIRs(global.get(colsFieldName), row.get("__values")) { case Seq(cols, values) =>
                row.drop("__values").insert(
                  "__explode" ->
                    ToArray(global.get("__old_col_idx").stream.streamFlatMap { oldColIndex =>
                      bindIR(cols.at(oldColIndex)) { col =>
                        values.stream.streamFlatMap { v =>
                          bindIR(v.get(entriesFieldName).at(oldColIndex)) { entry =>
                            val typ = child.typ
                            val rfields = typ.rowValueStruct.fieldNames.map(f => f -> v.get(f))
                            val cfields = typ.colType.fieldNames.map(f => f -> col.get(f))
                            val efields = typ.entryType.fieldNames.map(f => f -> entry.get(f))

                            val ordering =
                              typ.rowValueStruct.fieldNames ++
                                typ.colType.fieldNames ++
                                typ.entryType.fieldNames

                            val Rs = rfields.length
                            val Cs = cfields.length
                            val max = math.max(math.max(Rs, Cs), efields.length)

                            val newRow =
                              if (max == Rs) v
                                .select(typ.rowValueStruct.fieldNames)
                                .insert(cfields ++ efields, Some(ordering))
                              else if (max == Cs) col.insert(rfields ++ efields, Some(ordering))
                              else entry.insert(rfields ++ cfields, Some(ordering))

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
              row.get("__explode").bind { exploded =>
                val fields = x.typ.rowType.fieldNames

                val (rowNames, explodedNames) =
                  fields.partition(child.typ.rowKey.contains)

                if (rowNames.length > explodedNames.length)
                  row.select(rowNames).insert(
                    explodedNames.map(f => f -> exploded.get(f)),
                    Some(fields),
                  )
                else
                  exploded.select(explodedNames).insert(
                    rowNames.map(f => f -> row.get(f)),
                    Some(fields),
                  )
              }
            }
            .mapGlobals((g, _) => g.drop(colsFieldName, "__old_col_idx"))
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
              .mapGlobals((g, _) => g.drop(colsFieldName))

          if (child.typ.colKey.isEmpty) result
          else {
            assert(child.typ.rowKey.isEmpty)
            result.keyBy(child.typ.colKey)
          }
        }

      case MatrixToTableApply(child, function) =>
        val loweredChild = lower(ctx, child)
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
        lower(ctx, child)
          .mapGlobals((g, _) => g.drop(colsFieldName))
          .mapRows((_, row) => row.drop(entriesFieldName))

      case MatrixColsTable(child) =>
        val colKey = child.typ.colKey

        bindIR(lower(ctx, child).getGlobals) { global =>
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

      case table => lowerChildren(ctx, table).asInstanceOf[TableIR]
    }

    assertTypeUnchanged(tir, lowered)
    lowered
  }

  private def lower(ctx: ExecuteContext, bmir: BlockMatrixIR): BlockMatrixIR = {
    val lowered = lowerChildren(ctx, bmir).asInstanceOf[BlockMatrixIR]
    assertTypeUnchanged(bmir, lowered)
    lowered
  }

  private def lower(ctx: ExecuteContext, ir: IR): IR = {
    val lowered = ir match {
      case MatrixToValueApply(child, function) =>
        TableToValueApply(
          lower(ctx, child),
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
          lower(ctx, child),
          WrappedMatrixWriter(writer, colsFieldName, entriesFieldName, child.typ.colKey),
        )
      case MatrixMultiWrite(children, writer) =>
        TableMultiWrite(
          children.map(lower(ctx, _)),
          WrappedMatrixNativeMultiWriter(writer, children.head.typ.colKey),
        )
      case MatrixCount(child) =>
        lower(ctx, child).aggregate { (global, _) =>
          maketuple(ApplyAggOp(Count())(), global.get(colsFieldName).len)
        }
      case MatrixAggregate(child, query) =>
        lower(ctx, child).aggregate { (global, row) =>
          M.agg {
            for {
              cols <- global.get(colsFieldName)
              entries <- row.get(entriesFieldName)
              _ <- globalName -> global.select(child.typ.globalType.fieldNames)
              _ <- rowName -> row.select(child.typ.rowType.fieldNames)
              _ <- M.interleave(EVAL)(globalName -> global.select(child.typ.globalType.fieldNames))
            } yield cols
              .stream
              .zip(entries.stream)((c, e) => maybeIR(e)(e => maketuple(c, e)))
              .filter(!_.isNA)
              .aggExplode { explodedTuple =>
                M.agg {
                  M.sequence(
                    colName -> explodedTuple.get(0),
                    entryName -> explodedTuple.get(1),
                    query,
                  )
                }
              }
          }
        }
      case _ =>
        lowerChildren(ctx, ir).asInstanceOf[IR]
    }
    assertTypeUnchanged(ir, lowered)
    lowered
  }

  private def assertTypeUnchanged(original: BaseIR, lowered: BaseIR): Unit =
    if (lowered.typ != original.typ)
      fatal(
        s"lowering changed type:" +
          s"\n  before: ${original.typ}" +
          s"\n  after: ${lowered.typ}" +
          s"\n  ${original.getClass.getName} => ${lowered.getClass.getName}"
      )
}

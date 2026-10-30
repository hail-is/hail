package is.hail.expr.ir

import is.hail.collection.FastSeq
import is.hail.expr.ir.Scopes.ScopesSyntax

// One function per (node, child) whose env differs from its parent's. Each
// takes the env, then the node's name fields and the values it binds, in the
// order it binds them. A value bound in several scopes comes once.
object Binds {
  import Scope.{AGG, EVAL, REL, SCAN}

  // ApplyAggOp's and ApplyScanOp's init args, AggFold's zero,
  // AggArrayPerElement's knownLength, with s = AGG or SCAN: can't see s
  def aggInit[E[_]: Scopes, A](e: E[A], s: Scope): E[A] = e.drop(s)

  // ApplyAggOp's and ApplyScanOp's seq args, and the agg nodes' first child:
  // s's names become eval's, and the other of AGG and SCAN is gone. EVAL: e
  def aggSeq[E[_]: Scopes, A](e: E[A], s: Scope): E[A] =
    s match {
      case EVAL => e
      case AGG => e.promote(AGG).drop(SCAN)
      case SCAN => e.promote(SCAN).drop(AGG)
    }

  // the value of a Block binding in scope s; the name is bound by the caller
  def blockBinding[E[_]: Scopes, A](e: E[A], s: Scope): E[A] = aggSeq(e, s)

  // value nodes

  def tailLoop[E[_]: Scopes, A](e: E[A], name: Name, params: Seq[Name], args: Seq[A], recur: A)
    : E[A] =
    e.bind(EVAL, params.zip(args) :+ name -> recur)

  def streamMap[E[_]: Scopes, A](e: E[A], name: Name, elt: A): E[A] =
    e.bind(EVAL, FastSeq(name -> elt))

  def streamFor[E[_]: Scopes, A](e: E[A], name: Name, elt: A): E[A] =
    e.bind(EVAL, FastSeq(name -> elt))

  def streamFlatMap[E[_]: Scopes, A](e: E[A], name: Name, elt: A): E[A] =
    e.bind(EVAL, FastSeq(name -> elt))

  def streamFilter[E[_]: Scopes, A](e: E[A], name: Name, elt: A): E[A] =
    e.bind(EVAL, FastSeq(name -> elt))

  def streamTakeWhile[E[_]: Scopes, A](e: E[A], name: Name, elt: A): E[A] =
    e.bind(EVAL, FastSeq(name -> elt))

  def streamDropWhile[E[_]: Scopes, A](e: E[A], name: Name, elt: A): E[A] =
    e.bind(EVAL, FastSeq(name -> elt))

  // StreamBufferedAggregate's newKey and seqOps
  def streamBufferedAggregate[E[_]: Scopes, A](e: E[A], name: Name, elt: A): E[A] =
    e.bind(EVAL, FastSeq(name -> elt))

  // RunAggScan's seqs and result
  def runAggScan[E[_]: Scopes, A](e: E[A], name: Name, elt: A): E[A] =
    e.bind(EVAL, FastSeq(name -> elt))

  def ndArrayMap[E[_]: Scopes, A](e: E[A], name: Name, elt: A): E[A] =
    e.bind(EVAL, FastSeq(name -> elt))

  def streamZip[E[_]: Scopes, A](e: E[A], names: Seq[Name], elts: Seq[A]): E[A] =
    e.bind(EVAL, names.zip(elts))

  def streamZipJoin[E[_]: Scopes, A](e: E[A], curKey: Name, curVals: Name, key: A, vals: A): E[A] =
    e.bind(EVAL, FastSeq(curKey -> key, curVals -> vals))

  def streamZipJoinProducersMakeProducer[E[_]: Scopes, A](e: E[A], ctxName: Name, ctx: A): E[A] =
    e.bind(EVAL, FastSeq(ctxName -> ctx))

  def streamZipJoinProducersJoinF[E[_]: Scopes, A](
    e: E[A],
    curKey: Name,
    curVals: Name,
    key: A,
    vals: A,
  ): E[A] =
    e.bind(EVAL, FastSeq(curKey -> key, curVals -> vals))

  def streamLeftIntervalJoin[E[_]: Scopes, A](
    e: E[A],
    lEltName: Name,
    rEltName: Name,
    l: A,
    r: A,
  ): E[A] =
    e.bind(EVAL, FastSeq(lEltName -> l, rEltName -> r))

  def streamFold[E[_]: Scopes, A](e: E[A], accumName: Name, valueName: Name, acc: A, elt: A): E[A] =
    e.bind(EVAL, FastSeq(accumName -> acc, valueName -> elt))

  def streamScan[E[_]: Scopes, A](e: E[A], accumName: Name, valueName: Name, acc: A, elt: A): E[A] =
    e.bind(EVAL, FastSeq(accumName -> acc, valueName -> elt))

  def streamFold2Seq[E[_]: Scopes, A](
    e: E[A],
    valueName: Name,
    accumNames: Seq[Name],
    elt: A,
    accs: Seq[A],
  ): E[A] =
    e.bind(EVAL, (valueName -> elt) +: accumNames.zip(accs))

  def streamFold2Result[E[_]: Scopes, A](e: E[A], accumNames: Seq[Name], accs: Seq[A]): E[A] =
    e.bind(EVAL, accumNames.zip(accs))

  // StreamAgg's body: name bound in a new agg scope only; no scan scope
  def streamAgg[E[_]: Scopes, A](e: E[A], name: Name, elt: A): E[A] =
    e.drop(SCAN).create(AGG).bind(AGG, FastSeq(name -> elt))

  // StreamAggScan's body: name bound in a new scan scope and in eval; no agg
  def streamAggScan[E[_]: Scopes, A](e: E[A], name: Name, elt: A): E[A] =
    e.drop(AGG).create(SCAN).bind(SCAN, FastSeq(name -> elt)).bind(EVAL, FastSeq(name -> elt))

  def streamJoinRightDistinct[E[_]: Scopes, A](e: E[A], l: Name, r: Name, lElt: A, rElt: A): E[A] =
    e.bind(EVAL, FastSeq(l -> lElt, r -> rElt))

  def arraySort[E[_]: Scopes, A](e: E[A], left: Name, right: Name, l: A, r: A): E[A] =
    e.bind(EVAL, FastSeq(left -> l, right -> r))

  def arrayMaximalIndependentSet[E[_]: Scopes, A](e: E[A], left: Name, right: Name, l: A, r: A)
    : E[A] =
    e.fresh.bind(EVAL, FastSeq(left -> l, right -> r))

  // AggArrayPerElement's aggBody: elementName in s, indexName in s and eval
  def aggArrayPerElement[E[_]: Scopes, A](
    e: E[A],
    s: Scope,
    elementName: Name,
    indexName: Name,
    elt: A,
    idx: A,
  ): E[A] =
    e.bind(s, FastSeq(elementName -> elt, indexName -> idx)).bind(EVAL, FastSeq(indexName -> idx))

  def aggFoldSeqOp[E[_]: Scopes, A](e: E[A], s: Scope, accumName: Name, acc: A): E[A] =
    aggSeq(e, s).bind(EVAL, FastSeq(accumName -> acc))

  def aggFoldCombOp[E[_]: Scopes, A](
    e: E[A],
    accumName: Name,
    otherAccumName: Name,
    l: A,
    r: A,
  ): E[A] =
    e.fresh.bind(EVAL, FastSeq(accumName -> l, otherAccumName -> r))

  def aggExplode[E[_]: Scopes, A](e: E[A], s: Scope, name: Name, elt: A): E[A] =
    e.bind(s, FastSeq(name -> elt))

  def ndArrayMap2[E[_]: Scopes, A](e: E[A], lName: Name, rName: Name, l: A, r: A): E[A] =
    e.bind(EVAL, FastSeq(lName -> l, rName -> r))

  def collectDistributedArray[E[_]: Scopes, A](
    e: E[A],
    cname: Name,
    gname: Name,
    ctx: A,
    globals: A,
  ): E[A] =
    e.fresh.bind(EVAL, FastSeq(cname -> ctx, gname -> globals))

  // TableAggregate's query
  def tableAggregate[E[_]: Scopes, A](e: E[A], global: A, row: A): E[A] = {
    import TableIR.{globalName, rowName}
    e.fresh
      .create(AGG)
      .bind(AGG, FastSeq(globalName -> global, rowName -> row))
      .bind(EVAL, FastSeq(globalName -> global))
  }

  // MatrixAggregate's query
  def matrixAggregate[E[_]: Scopes, A](e: E[A], global: A, sa: A, va: A, g: A): E[A] = {
    import MatrixIR.{colName, entryName, globalName, rowName}
    e.fresh
      .create(AGG)
      .bind(AGG, FastSeq(globalName -> global, colName -> sa, rowName -> va, entryName -> g))
      .bind(EVAL, FastSeq(globalName -> global))
  }

  // RelationalLet's body: name bound in REL; no agg or scan scope
  def relationalLet[E[_]: Scopes, A](e: E[A], name: Name, v: A): E[A] =
    e.bind(REL, FastSeq(name -> v)).drop(AGG).drop(SCAN)

  // table nodes

  def tableFilter[E[_]: Scopes, A](e: E[A], global: A, row: A): E[A] =
    e.fresh.bind(EVAL, FastSeq(TableIR.globalName -> global, TableIR.rowName -> row))

  def tableGen[E[_]: Scopes, A](e: E[A], cname: Name, gname: Name, ctx: A, globals: A): E[A] =
    e.fresh.bind(EVAL, FastSeq(cname -> ctx, gname -> globals))

  def tableMapGlobals[E[_]: Scopes, A](e: E[A], global: A): E[A] =
    e.fresh.bind(EVAL, FastSeq(TableIR.globalName -> global))

  // TableMapRows' newRow
  def tableMapRows[E[_]: Scopes, A](e: E[A], global: A, row: A): E[A] = {
    import TableIR.{globalName, rowName}
    e.fresh
      .create(SCAN)
      .bind(SCAN, FastSeq(globalName -> global, rowName -> row))
      .bind(EVAL, FastSeq(globalName -> global, rowName -> row))
  }

  def tableAggregateByKey[E[_]: Scopes, A](e: E[A], global: A, row: A): E[A] = {
    import TableIR.{globalName, rowName}
    e.fresh
      .create(AGG)
      .bind(AGG, FastSeq(globalName -> global, rowName -> row))
      .bind(EVAL, FastSeq(globalName -> global))
  }

  def tableKeyByAndAggregateExpr[E[_]: Scopes, A](e: E[A], global: A, row: A): E[A] = {
    import TableIR.{globalName, rowName}
    e.fresh
      .create(AGG)
      .bind(AGG, FastSeq(globalName -> global, rowName -> row))
      .bind(EVAL, FastSeq(globalName -> global))
  }

  def tableKeyByAndAggregateNewKey[E[_]: Scopes, A](e: E[A], global: A, row: A): E[A] =
    e.fresh.bind(EVAL, FastSeq(TableIR.globalName -> global, TableIR.rowName -> row))

  def tableMapPartitions[E[_]: Scopes, A](
    e: E[A],
    globalName: Name,
    partitionStreamName: Name,
    global: A,
    partition: A,
  ): E[A] =
    e.fresh.bind(EVAL, FastSeq(globalName -> global, partitionStreamName -> partition))

  def relationalLetTable[E[_]: Scopes, A](e: E[A], name: Name, v: A): E[A] =
    e.fresh.bind(REL, FastSeq(name -> v))

  // matrix nodes

  // MatrixMapRows' newRow: agg ranges over this row's entries
  def matrixMapRows[E[_]: Scopes, A](e: E[A], global: A, sa: A, va: A, g: A, nCols: A): E[A] = {
    import MatrixIR.{colName, entryName, globalName, rowName}
    e.fresh
      .create(AGG)
      .create(SCAN)
      .bind(AGG, FastSeq(globalName -> global, colName -> sa, rowName -> va, entryName -> g))
      .bind(SCAN, FastSeq(globalName -> global, rowName -> va))
      .bind(EVAL, FastSeq(globalName -> global, rowName -> va, Name("n_cols") -> nCols))
  }

  // MatrixMapCols' newCol: agg ranges over this col's entries
  def matrixMapCols[E[_]: Scopes, A](e: E[A], global: A, sa: A, va: A, g: A, nRows: A): E[A] = {
    import MatrixIR.{colName, entryName, globalName, rowName}
    e.fresh
      .create(AGG)
      .create(SCAN)
      .bind(AGG, FastSeq(globalName -> global, colName -> sa, rowName -> va, entryName -> g))
      .bind(SCAN, FastSeq(globalName -> global, colName -> sa))
      .bind(EVAL, FastSeq(globalName -> global, colName -> sa, Name("n_rows") -> nRows))
  }

  def matrixFilterRows[E[_]: Scopes, A](e: E[A], global: A, va: A): E[A] =
    e.fresh.bind(EVAL, FastSeq(MatrixIR.globalName -> global, MatrixIR.rowName -> va))

  def matrixFilterCols[E[_]: Scopes, A](e: E[A], global: A, sa: A): E[A] =
    e.fresh.bind(EVAL, FastSeq(MatrixIR.globalName -> global, MatrixIR.colName -> sa))

  def matrixMapEntries[E[_]: Scopes, A](e: E[A], global: A, sa: A, va: A, g: A): E[A] = {
    import MatrixIR.{colName, entryName, globalName, rowName}
    e.fresh.bind(
      EVAL,
      FastSeq(globalName -> global, colName -> sa, rowName -> va, entryName -> g),
    )
  }

  def matrixFilterEntries[E[_]: Scopes, A](e: E[A], global: A, sa: A, va: A, g: A): E[A] = {
    import MatrixIR.{colName, entryName, globalName, rowName}
    e.fresh.bind(
      EVAL,
      FastSeq(globalName -> global, colName -> sa, rowName -> va, entryName -> g),
    )
  }

  def matrixMapGlobals[E[_]: Scopes, A](e: E[A], global: A): E[A] =
    e.fresh.bind(EVAL, FastSeq(MatrixIR.globalName -> global))

  // MatrixAggregateRowsByKey's entryExpr: agg over the key's entries in a col
  def matrixAggregateRowsByKeyEntryExpr[E[_]: Scopes, A](e: E[A], global: A, sa: A, va: A, g: A)
    : E[A] = {
    import MatrixIR.{colName, entryName, globalName, rowName}
    e.fresh
      .create(AGG)
      .bind(AGG, FastSeq(globalName -> global, colName -> sa, rowName -> va, entryName -> g))
      .bind(EVAL, FastSeq(globalName -> global, colName -> sa))
  }

  // MatrixAggregateRowsByKey's rowExpr: agg ranges over the key's rows
  def matrixAggregateRowsByKeyRowExpr[E[_]: Scopes, A](e: E[A], global: A, va: A): E[A] = {
    import MatrixIR.{globalName, rowName}
    e.fresh
      .create(AGG)
      .bind(AGG, FastSeq(globalName -> global, rowName -> va))
      .bind(EVAL, FastSeq(globalName -> global))
  }

  // MatrixAggregateColsByKey's entryExpr: agg over the key's entries in a row
  def matrixAggregateColsByKeyEntryExpr[E[_]: Scopes, A](e: E[A], global: A, sa: A, va: A, g: A)
    : E[A] = {
    import MatrixIR.{colName, entryName, globalName, rowName}
    e.fresh
      .create(AGG)
      .bind(AGG, FastSeq(globalName -> global, colName -> sa, rowName -> va, entryName -> g))
      .bind(EVAL, FastSeq(globalName -> global, rowName -> va))
  }

  // MatrixAggregateColsByKey's colExpr: agg ranges over the key's cols
  def matrixAggregateColsByKeyColExpr[E[_]: Scopes, A](e: E[A], global: A, sa: A): E[A] = {
    import MatrixIR.{colName, globalName}
    e.fresh
      .create(AGG)
      .bind(AGG, FastSeq(globalName -> global, colName -> sa))
      .bind(EVAL, FastSeq(globalName -> global))
  }

  def relationalLetMatrixTable[E[_]: Scopes, A](e: E[A], name: Name, v: A): E[A] =
    e.fresh.bind(REL, FastSeq(name -> v))

  // block matrix nodes

  def blockMatrixMap[E[_]: Scopes, A](e: E[A], eltName: Name, elt: A): E[A] =
    e.fresh.bind(EVAL, FastSeq(eltName -> elt))

  def blockMatrixMap2[E[_]: Scopes, A](e: E[A], leftName: Name, rightName: Name, l: A, r: A): E[A] =
    e.fresh.bind(EVAL, FastSeq(leftName -> l, rightName -> r))
}

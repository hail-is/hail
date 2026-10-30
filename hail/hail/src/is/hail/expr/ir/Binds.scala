package is.hail.expr.ir

import is.hail.collection.FastSeq
import is.hail.expr.ir.Scopes.ScopesSyntax
import is.hail.expr.ir.defs._
import is.hail.types.tcoerce
import is.hail.types.virtual._
import is.hail.types.virtual.TIterable.elementType

import scala.collection.immutable.ArraySeq

sealed abstract class AggEnv {
  def empty: AggEnv = this match {
    case AggEnv.Create(_) => AggEnv.Create(Seq.empty)
    case AggEnv.Bind(_) => AggEnv.NoOp
    case AggEnv.NoOp => AggEnv.NoOp
    case AggEnv.Drop => AggEnv.Drop
    case AggEnv.Promote => AggEnv.Promote
  }

  def isEmpty: Boolean = this match {
    case a: AggEnv.Modify => a.bindings.isEmpty
    case _ => true
  }
}

object AggEnv {
  case object NoOp extends AggEnv
  case object Drop extends AggEnv
  case object Promote extends AggEnv

  abstract class Modify extends AggEnv { def bindings: Seq[Int] }
  final case class Create(override val bindings: Seq[Int]) extends Modify
  final case class Bind(override val bindings: Seq[Int]) extends Modify

  def bindOrNoOp(bindings: Seq[Int]): AggEnv =
    if (bindings.nonEmpty) Bind(bindings) else NoOp
}

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

final case class Bindings[+T](
  all: IndexedSeq[(Name, T)],
  eval: IndexedSeq[Int],
  agg: AggEnv,
  scan: AggEnv,
  relational: IndexedSeq[Int],
  dropEval: Boolean,
) {
  def map[U](f: (Name, T) => U): Bindings[U] =
    copy(all = all.map { case (n, t) => (n, f(n, t)) })

  def allEmpty: Boolean =
    eval.isEmpty && agg.isEmpty && scan.isEmpty && relational.isEmpty

  def dropBindings[U]: Bindings[U] =
    new Bindings(FastSeq(), FastSeq(), agg.empty, scan.empty, FastSeq(), dropEval)
}

object Bindings {
  def apply[T](
    bindings: IndexedSeq[(Name, T)] = FastSeq.empty,
    eval: IndexedSeq[Int] = FastSeq.empty,
    agg: AggEnv = AggEnv.NoOp,
    scan: AggEnv = AggEnv.NoOp,
    relational: IndexedSeq[Int] = FastSeq.empty,
    dropEval: Boolean = false,
  ): Bindings[T] =
    new Bindings(bindings, eval, agg, scan, relational, dropEval)

  def inEval[T](bindings: IndexedSeq[(Name, T)]): Bindings[T] =
    Bindings(bindings, eval = bindings.indices)

  val empty: Bindings[Nothing] =
    new Bindings(FastSeq(), FastSeq(), AggEnv.NoOp, AggEnv.NoOp, FastSeq(), dropEval = false)

  /** Returns the environment of the `i`th child of `ir` given the environment of the parent node
    * `ir`.
    */
  def get(ir: BaseIR, i: Int): Bindings[Type] =
    ir match {
      case ir: MatrixIR => childEnvMatrix(ir, i)
      case ir: TableIR => childEnvTable(ir, i)
      case ir: BlockMatrixIR => childEnvBlockMatrix(ir, i)
      case ir: IR => childEnvValue(ir, i)
    }

  // As BindingEnv.extend: a fresh scope, then agg's and scan's create, promote or drop, then
  // the binds. A scope is bound only by Create or Bind, since binding a missing scope fails.
  private[ir] def applyRecord[E[_]: Scopes, A](e: E[A], b: Bindings[A]): E[A] = {
    def change(e: E[A], s: Scope, a: AggEnv): E[A] =
      a match {
        case AggEnv.Drop => e.drop(s)
        case AggEnv.Promote => e.promote(s)
        case AggEnv.Create(_) => e.create(s)
        case _ => e
      }

    def bindIn(e: E[A], s: Scope, a: AggEnv): E[A] =
      a match {
        case a: AggEnv.Modify => e.bind(s, a.bindings.map(b.all))
        case _ => e
      }

    val e1 = if (b.dropEval) e.fresh else e
    val e2 = change(change(e1, Scope.AGG, b.agg), Scope.SCAN, b.scan)
    val e3 = if (b.eval.isEmpty) e2 else e2.bind(Scope.EVAL, b.eval.map(b.all))
    val e4 = bindIn(bindIn(e3, Scope.AGG, b.agg), Scope.SCAN, b.scan)
    if (b.relational.isEmpty) e4 else e4.bind(Scope.REL, b.relational.map(b.all))
  }

  // Create a `Bindings` which cannot see anything bound in the enclosing context.
  private def inFreshScope(
    bindings: IndexedSeq[(Name, Type)] = FastSeq.empty,
    eval: IndexedSeq[Int] = FastSeq.empty,
    agg: Option[IndexedSeq[Int]] = None,
    scan: Option[IndexedSeq[Int]] = None,
    relational: IndexedSeq[Int] = FastSeq.empty,
  ): Bindings[Type] =
    new Bindings(
      bindings,
      if (eval.isEmpty && agg.isEmpty && scan.isEmpty && relational.isEmpty) bindings.indices
      else eval,
      agg.map(AggEnv.Create(_)).getOrElse(AggEnv.Drop),
      scan.map(AggEnv.Create(_)).getOrElse(AggEnv.Drop),
      relational,
      dropEval = true,
    )

  private def childEnvMatrix(ir: MatrixIR, i: Int): Bindings[Type] = {
    import is.hail.types.virtual.MatrixType.{
      globalBindings, rowInEntryBindings, colInColBindings, rowInRowBindings, entryBindings,
      colInEntryBindings,
    }
    ir match {
      case MatrixMapRows(child, _) if i == 1 =>
        Bindings.inFreshScope(
          child.typ.entryBindings :+ Name("n_cols") -> TInt32,
          eval = rowInEntryBindings :+ 4,
          agg = Some(entryBindings),
          scan = Some(rowInEntryBindings),
        )
      case MatrixFilterRows(child, _) if i == 1 =>
        Bindings.inFreshScope(child.typ.rowBindings)
      case MatrixMapCols(child, _, _) if i == 1 =>
        Bindings.inFreshScope(
          child.typ.entryBindings :+ Name("n_rows") -> TInt64,
          eval = colInEntryBindings :+ 4,
          agg = Some(entryBindings),
          scan = Some(colInEntryBindings),
        )
      case MatrixFilterCols(child, _) if i == 1 =>
        Bindings.inFreshScope(child.typ.colBindings)
      case MatrixMapEntries(child, _) if i == 1 =>
        Bindings.inFreshScope(child.typ.entryBindings)
      case MatrixFilterEntries(child, _) if i == 1 =>
        Bindings.inFreshScope(child.typ.entryBindings)
      case MatrixMapGlobals(child, _) if i == 1 =>
        Bindings.inFreshScope(child.typ.globalBindings)
      case MatrixAggregateColsByKey(child, _, _) =>
        if (i == 1)
          Bindings.inFreshScope(
            child.typ.entryBindings,
            eval = rowInEntryBindings,
            agg = Some(entryBindings),
          )
        else if (i == 2)
          Bindings.inFreshScope(
            child.typ.colBindings,
            eval = globalBindings,
            agg = Some(colInColBindings),
          )
        else
          Bindings.inFreshScope()
      case MatrixAggregateRowsByKey(child, _, _) =>
        if (i == 1)
          Bindings.inFreshScope(
            child.typ.entryBindings,
            eval = colInEntryBindings,
            agg = Some(entryBindings),
          )
        else if (i == 2)
          Bindings.inFreshScope(
            child.typ.rowBindings,
            eval = globalBindings,
            agg = Some(rowInRowBindings),
          )
        else
          Bindings.inFreshScope()
      case RelationalLetMatrixTable(name, value, _) if i == 1 =>
        Bindings.inFreshScope(FastSeq(name -> value.typ), relational = FastSeq(0))
      case _ =>
        Bindings.inFreshScope()
    }
  }

  private def childEnvTable(ir: TableIR, i: Int): Bindings[Type] = {
    import is.hail.types.virtual.TableType.{globalBindings, rowBindings}
    ir match {
      case TableFilter(child, _) if i == 1 =>
        Bindings.inFreshScope(child.typ.rowBindings)
      case TableGen(contexts, globals, cname, gname, _, _, _) if i == 2 =>
        Bindings.inFreshScope(FastSeq(
          cname -> elementType(contexts.typ),
          gname -> globals.typ,
        ))
      case TableMapGlobals(child, _) if i == 1 =>
        Bindings.inFreshScope(child.typ.globalBindings)
      case TableMapRows(child, _) if i == 1 =>
        Bindings.inFreshScope(child.typ.rowBindings, eval = rowBindings, scan = Some(rowBindings))
      case TableAggregateByKey(child, _) if i == 1 =>
        Bindings.inFreshScope(child.typ.rowBindings, eval = globalBindings, agg = Some(rowBindings))
      case TableKeyByAndAggregate(child, _, _, _, _) =>
        if (i == 1)
          Bindings.inFreshScope(
            child.typ.rowBindings,
            eval = globalBindings,
            agg = Some(rowBindings),
          )
        else if (i == 2)
          Bindings.inFreshScope(child.typ.rowBindings)
        else Bindings.inFreshScope()
      case TableMapPartitions(child, g, p, _, _, _) if i == 1 =>
        Bindings.inFreshScope(FastSeq(
          g -> child.typ.globalType,
          p -> TStream(child.typ.rowType),
        ))
      case RelationalLetTable(name, value, _) if i == 1 =>
        Bindings.inFreshScope(FastSeq(name -> value.typ), relational = FastSeq(0))
      case _ =>
        Bindings.inFreshScope()
    }
  }

  private def childEnvBlockMatrix(ir: BlockMatrixIR, i: Int): Bindings[Type] = {
    ir match {
      case BlockMatrixMap(_, eltName, _, _) if i == 1 =>
        Bindings.inFreshScope(FastSeq(eltName -> TFloat64))
      case BlockMatrixMap2(_, _, lName, rName, _, _) if i == 2 =>
        Bindings.inFreshScope(FastSeq(lName -> TFloat64, rName -> TFloat64))
      case _ =>
        Bindings.inFreshScope()
    }
  }

  private def childEnvValue(ir: IR, i: Int): Bindings[Type] =
    ir match {
      case Block(bindings, _) =>
        val types = ArraySeq.newBuilder[(Name, Type)]
        types.sizeHint(i)

        val eval = ArraySeq.newBuilder[Int]
        eval.sizeHint(i) // most likely binding in eval

        val agg = ArraySeq.newBuilder[Int]
        val scan = ArraySeq.newBuilder[Int]

        for (k <- 0 until i) {
          val Binding(name, value, scope) = bindings(k)
          types += name -> value.typ
          scope match {
            case Scope.EVAL =>
              eval += k
            case Scope.AGG =>
              agg += k
            case Scope.SCAN =>
              scan += k
          }
        }

        if (i == bindings.length || bindings(i).scope == Scope.EVAL)
          Bindings(
            types.result(),
            eval.result(),
            AggEnv.bindOrNoOp(agg.result()),
            AggEnv.bindOrNoOp(scan.result()),
          )
        else if (bindings(i).scope == Scope.AGG)
          Bindings(types.result(), agg.result(), AggEnv.Promote, AggEnv.Drop)
        else // SCAN
          Bindings(types.result(), scan.result(), AggEnv.Drop, AggEnv.Promote)

      case TailLoop(name, args, resultType, _) if i == args.length =>
        Bindings.inEval(
          args.map { case (name, ir) => name -> ir.typ } :+
            name -> TTuple(TTuple(args.map(_._2.typ): _*), resultType)
        )
      case StreamMap(a, name, _) if i == 1 =>
        Bindings.inEval(FastSeq(name -> elementType(a.typ)))
      case StreamZip(as, names, _, _, _) if i == as.length =>
        Bindings.inEval(names.zip(as.map(a => elementType(a.typ))))
      case StreamZipJoin(as, key, curKey, curVals, _) if i == as.length =>
        val eltType = tcoerce[TStruct](elementType(as.head.typ))
        Bindings.inEval(FastSeq(
          curKey -> eltType.typeAfterSelectNames(key),
          curVals -> TArray(eltType),
        ))
      case StreamZipJoinProducers(contexts, ctxName, makeProducer, key, curKey, curVals, _) =>
        if (i == 1) {
          val contextType = elementType(contexts.typ)
          Bindings.inEval(FastSeq(ctxName -> contextType))
        } else if (i == 2) {
          val eltType = tcoerce[TStruct](elementType(makeProducer.typ))
          Bindings.inEval(FastSeq(
            curKey -> eltType.typeAfterSelectNames(key),
            curVals -> TArray(eltType),
          ))
        } else Bindings.empty
      case StreamLeftIntervalJoin(left, right, _, _, lEltName, rEltName, _) if i == 2 =>
        Bindings.inEval(FastSeq(
          lEltName -> elementType(left.typ),
          rEltName -> TArray(elementType(right.typ)),
        ))
      case StreamFor(a, name, _) if i == 1 =>
        Bindings.inEval(FastSeq(name -> elementType(a.typ)))
      case StreamFlatMap(a, name, _) if i == 1 =>
        Bindings.inEval(FastSeq(name -> elementType(a.typ)))
      case StreamFilter(a, name, _) if i == 1 =>
        Bindings.inEval(FastSeq(name -> elementType(a.typ)))
      case StreamTakeWhile(a, name, _) if i == 1 =>
        Bindings.inEval(FastSeq(name -> elementType(a.typ)))
      case StreamDropWhile(a, name, _) if i == 1 =>
        Bindings.inEval(FastSeq(name -> elementType(a.typ)))
      case StreamFold(a, zero, accumName, valueName, _) if i == 2 =>
        Bindings.inEval(FastSeq(accumName -> zero.typ, valueName -> elementType(a.typ)))
      case StreamFold2(a, accum, valueName, _, _) =>
        if (i < accum.length + 1)
          Bindings.empty
        else if (i < 2 * accum.length + 1)
          Bindings.inEval(
            (valueName -> elementType(a.typ)) +:
              accum.map { case (name, value) => (name, value.typ) }
          )
        else
          Bindings.inEval(accum.map { case (name, value) => (name, value.typ) })
      case StreamBufferedAggregate(stream, _, _, _, name, _, _) if i >= 2 =>
        Bindings.inEval(FastSeq(name -> elementType(stream.typ)))
      case RunAggScan(a, name, _, _, _, _) if i == 2 || i == 3 =>
        Bindings.inEval(FastSeq(name -> elementType(a.typ)))
      case StreamAgg(a, name, _) if i == 1 =>
        Bindings(
          FastSeq(name -> elementType(a.typ)),
          agg = AggEnv.Create(FastSeq(0)),
          scan = AggEnv.Drop,
        )
      case StreamScan(a, zero, accumName, valueName, _) if i == 2 =>
        Bindings.inEval(FastSeq(accumName -> zero.typ, valueName -> elementType(a.typ)))
      case StreamAggScan(a, name, _) if i == 1 =>
        val eltType = elementType(a.typ)
        Bindings(
          FastSeq(name -> eltType),
          eval = FastSeq(0),
          agg = AggEnv.Drop,
          scan = AggEnv.Create(FastSeq(0)),
        )
      case StreamJoinRightDistinct(ll, rr, _, _, l, r, _, _) if i == 2 =>
        Bindings.inEval(FastSeq(
          l -> elementType(ll.typ),
          r -> elementType(rr.typ),
        ))
      case ArraySort(a, left, right, _) if i == 1 =>
        Bindings.inEval(FastSeq(
          left -> elementType(a.typ),
          right -> elementType(a.typ),
        ))
      case ArrayMaximalIndependentSet(a, Some((left, right, _))) if i == 1 =>
        val typ = tcoerce[TBaseStruct](elementType(a.typ)).types.head
        val tupleType = TTuple(typ)
        Bindings.inFreshScope(FastSeq(left -> tupleType, right -> tupleType))
      case AggArrayPerElement(a, elementName, indexName, _, _, isScan) =>
        if (i == 0)
          Bindings(
            agg = if (isScan) AggEnv.Drop else AggEnv.Promote,
            scan = if (isScan) AggEnv.Promote else AggEnv.Drop,
          )
        else if (i == 1) {
          Bindings(
            FastSeq(
              elementName -> elementType(a.typ),
              indexName -> TInt32,
            ),
            eval = FastSeq(1),
            agg = if (isScan) AggEnv.NoOp else AggEnv.Bind(FastSeq(0, 1)),
            scan = if (isScan) AggEnv.Bind(FastSeq(0, 1)) else AggEnv.NoOp,
          )
        } else
          Bindings(
            agg = if (isScan) AggEnv.NoOp else AggEnv.Drop,
            scan = if (isScan) AggEnv.Drop else AggEnv.NoOp,
          )
      case AggFold(zero, _, _, accumName, otherAccumName, isScan) =>
        if (i == 0)
          Bindings(
            agg = if (isScan) AggEnv.NoOp else AggEnv.Drop,
            scan = if (isScan) AggEnv.Drop else AggEnv.NoOp,
          )
        else if (i == 1)
          Bindings(
            FastSeq(accumName -> zero.typ),
            eval = FastSeq(0),
            agg = if (isScan) AggEnv.Drop else AggEnv.Promote,
            scan = if (isScan) AggEnv.Promote else AggEnv.Drop,
          )
        else
          Bindings.inFreshScope(FastSeq(accumName -> zero.typ, otherAccumName -> zero.typ))
      case NDArrayMap(nd, name, _) if i == 1 =>
        Bindings.inEval(FastSeq(name -> tcoerce[TNDArray](nd.typ).elementType))
      case NDArrayMap2(l, r, lName, rName, _, _) if i == 2 =>
        Bindings.inEval(FastSeq(
          lName -> tcoerce[TNDArray](l.typ).elementType,
          rName -> tcoerce[TNDArray](r.typ).elementType,
        ))
      case CollectDistributedArray(contexts, globals, cname, gname, _, _, _, _) if i == 2 =>
        Bindings.inFreshScope(
          FastSeq(
            cname -> elementType(contexts.typ),
            gname -> globals.typ,
          )
        )
      case TableAggregate(child, _) =>
        if (i == 0) Bindings.inFreshScope()
        else Bindings.inFreshScope(
          child.typ.rowBindings,
          eval = TableType.globalBindings,
          agg = Some(TableType.rowBindings),
        )
      case MatrixAggregate(child, _) =>
        if (i == 0) Bindings.inFreshScope()
        else Bindings.inFreshScope(
          child.typ.entryBindings,
          eval = MatrixType.globalBindings,
          agg = Some(MatrixType.entryBindings),
        )
      case ApplyAggOp(init, _, _) =>
        if (i < init.length) Bindings(agg = AggEnv.Drop)
        else Bindings(agg = AggEnv.Promote, scan = AggEnv.Drop)
      case ApplyScanOp(init, _, _) =>
        if (i < init.length) Bindings(scan = AggEnv.Drop)
        else Bindings(agg = AggEnv.Drop, scan = AggEnv.Promote)
      case AggFilter(_, _, isScan) if i == 0 =>
        Bindings(
          agg = if (isScan) AggEnv.Drop else AggEnv.Promote,
          scan = if (isScan) AggEnv.Promote else AggEnv.Drop,
        )
      case AggGroupBy(_, _, isScan) if i == 0 =>
        Bindings(
          agg = if (isScan) AggEnv.Drop else AggEnv.Promote,
          scan = if (isScan) AggEnv.Promote else AggEnv.Drop,
        )
      case AggExplode(a, name, _, isScan) =>
        if (i == 0)
          Bindings(
            agg = if (isScan) AggEnv.Drop else AggEnv.Promote,
            scan = if (isScan) AggEnv.Promote else AggEnv.Drop,
          )
        else
          Bindings(
            FastSeq(name -> elementType(a.typ)),
            agg = if (isScan) AggEnv.NoOp else AggEnv.Bind(FastSeq(0)),
            scan = if (isScan) AggEnv.Bind(FastSeq(0)) else AggEnv.NoOp,
          )
      case RelationalLet(name, value, _) =>
        if (i == 0) Bindings.inFreshScope()
        else Bindings(
          FastSeq(name -> value.typ),
          agg = AggEnv.Drop,
          scan = AggEnv.Drop,
          relational = FastSeq(0),
        )
      case _ =>
        Bindings.empty
    }
}

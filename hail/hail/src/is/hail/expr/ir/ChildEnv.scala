package is.hail.expr.ir

import is.hail.collection.FastSeq
import is.hail.expr.ir.Scopes.ScopesSyntax
import is.hail.expr.ir.defs._
import is.hail.types.tcoerce
import is.hail.types.virtual._
import is.hail.types.virtual.TIterable.elementType

import cats.Functor
import cats.syntax.functor._

// One function per node whose children's envs differ from its own. Each takes the env, the
// child's index i, the node's name fields and its children, and returns child i's env. It reads
// types only from children(0 until i), makes each value with v, then calls the node's Binds rule.
object ChildEnv {
  import Scope.{AGG, SCAN}

  /** Returns the env of the `i`th child of `ir` given the env `e` of `ir`. `v` makes the value each
    * name is bound to.
    */
  def get[E[_]: Scopes, A](e: E[A], ir: BaseIR, i: Int)(v: (Name, Type) => A): E[A] =
    get(e, ir, ir.children.toIndexedSeq, i)(v)

  // cs may differ from ir's children; ir supplies only name fields
  def get[E[_]: Scopes, A](
    e: E[A],
    ir: BaseIR,
    cs: IndexedSeq[BaseIR],
    i: Int,
  )(
    v: (Name, Type) => A
  ): E[A] =
    ir match {
      case x: Block => block(e, i, x.bindings, cs)(v)
      case x: TailLoop => tailLoop(e, i, x.name, x.params.map(_._1), x.resultType, cs)(v)
      case x: StreamMap => streamMap(e, i, x.name, cs)(v)
      case x: StreamZip => streamZip(e, i, x.names, cs)(v)
      case x: StreamZipJoin =>
        streamZipJoin(e, i, x.as.length, x.key, x.curKey, x.curVals, cs)(v)
      case x: StreamZipJoinProducers =>
        streamZipJoinProducers(e, i, x.ctxName, x.key, x.curKey, x.curVals, cs)(v)
      case x: StreamLeftIntervalJoin => streamLeftIntervalJoin(e, i, x.lname, x.rname, cs)(v)
      case x: StreamFor => streamFor(e, i, x.valueName, cs)(v)
      case x: StreamFlatMap => streamFlatMap(e, i, x.name, cs)(v)
      case x: StreamFilter => streamFilter(e, i, x.name, cs)(v)
      case x: StreamTakeWhile => streamTakeWhile(e, i, x.elementName, cs)(v)
      case x: StreamDropWhile => streamDropWhile(e, i, x.elementName, cs)(v)
      case x: StreamFold => streamFold(e, i, x.accumName, x.valueName, cs)(v)
      case x: StreamFold2 => streamFold2(e, i, x.accum.map(_._1), x.valueName, cs)(v)
      case x: StreamBufferedAggregate => streamBufferedAggregate(e, i, x.name, cs)(v)
      case x: RunAggScan => runAggScan(e, i, x.name, cs)(v)
      case x: StreamAgg => streamAgg(e, i, x.name, cs)(v)
      case x: StreamScan => streamScan(e, i, x.accumName, x.valueName, cs)(v)
      case x: StreamAggScan => streamAggScan(e, i, x.name, cs)(v)
      case x: StreamJoinRightDistinct => streamJoinRightDistinct(e, i, x.l, x.r, cs)(v)
      case x: ArraySort => arraySort(e, i, x.left, x.right, cs)(v)
      case x: ArrayMaximalIndependentSet =>
        arrayMaximalIndependentSet(e, i, x.tieBreaker.map { case (l, r, _) => (l, r) }, cs)(v)
      case x: AggArrayPerElement =>
        aggArrayPerElement(e, i, x.elementName, x.indexName, x.isScan, cs)(v)
      case x: AggFold => aggFold(e, i, x.accumName, x.otherAccumName, x.isScan, cs)(v)
      case x: NDArrayMap => ndArrayMap(e, i, x.valueName, cs)(v)
      case x: NDArrayMap2 => ndArrayMap2(e, i, x.lName, x.rName, cs)(v)
      case x: CollectDistributedArray => collectDistributedArray(e, i, x.cname, x.gname, cs)(v)
      case _: TableAggregate => tableAggregate(e, i, cs)(v)
      case _: MatrixAggregate => matrixAggregate(e, i, cs)(v)
      case x: ApplyAggOp => applyAggOp(e, i, x.initOpArgs.length)
      case x: ApplyScanOp => applyScanOp(e, i, x.initOpArgs.length)
      case x: AggFilter => aggFilter(e, i, x.isScan)
      case x: AggGroupBy => aggGroupBy(e, i, x.isScan)
      case x: AggExplode => aggExplode(e, i, x.name, x.isScan, cs)(v)
      case x: RelationalLet => relationalLet(e, i, x.name, cs)(v)

      case _: TableFilter => tableFilter(e, i, cs)(v)
      case x: TableGen => tableGen(e, i, x.cname, x.gname, cs)(v)
      case _: TableMapGlobals => tableMapGlobals(e, i, cs)(v)
      case _: TableMapRows => tableMapRows(e, i, cs)(v)
      case _: TableAggregateByKey => tableAggregateByKey(e, i, cs)(v)
      case _: TableKeyByAndAggregate => tableKeyByAndAggregate(e, i, cs)(v)
      case x: TableMapPartitions =>
        tableMapPartitions(e, i, x.globalName, x.partitionStreamName, cs)(v)
      case x: RelationalLetTable => relationalLetTable(e, i, x.name, cs)(v)

      case _: MatrixMapRows => matrixMapRows(e, i, cs)(v)
      case _: MatrixFilterRows => matrixFilterRows(e, i, cs)(v)
      case _: MatrixMapCols => matrixMapCols(e, i, cs)(v)
      case _: MatrixFilterCols => matrixFilterCols(e, i, cs)(v)
      case _: MatrixMapEntries => matrixMapEntries(e, i, cs)(v)
      case _: MatrixFilterEntries => matrixFilterEntries(e, i, cs)(v)
      case _: MatrixMapGlobals => matrixMapGlobals(e, i, cs)(v)
      case _: MatrixAggregateColsByKey => matrixAggregateColsByKey(e, i, cs)(v)
      case _: MatrixAggregateRowsByKey => matrixAggregateRowsByKey(e, i, cs)(v)
      case x: RelationalLetMatrixTable => relationalLetMatrixTable(e, i, x.name, cs)(v)

      case x: BlockMatrixMap => blockMatrixMap(e, i, x.eltName)(v)
      case x: BlockMatrixMap2 => blockMatrixMap2(e, i, x.leftName, x.rightName)(v)

      case _: TableIR | _: MatrixIR | _: BlockMatrixIR => e.fresh
      case _ => e
    }

  // k returns child i's final form, whose type a Block binding binds
  def step[F[_]: Functor, E[_]: Scopes, A](
    e: E[A],
    ir: BaseIR,
    cs: IndexedSeq[BaseIR],
    i: Int,
  )(
    v: (Name, Type) => A
  )(
    k: E[A] => F[BaseIR]
  ): F[E[A]] =
    ir match {
      case x: Block =>
        val bs = x.bindings
        k(blockChild(e, i, bs)).map(c => if (i < bs.length) blockBind(e, bs(i), c)(v) else e)
      case _ =>
        k(get(e, ir, cs, i)(v)).as(e)
    }

  private def typ(c: BaseIR): Type = c.asInstanceOf[IR].typ
  private def eltType(c: BaseIR): Type = elementType(typ(c))
  private def tableType(c: BaseIR): TableType = c.asInstanceOf[TableIR].typ
  private def matrixType(c: BaseIR): MatrixType = c.asInstanceOf[MatrixIR].typ

  private def aggScope(isScan: Boolean): Scope = if (isScan) SCAN else AGG

  private def tableGlobal[A](v: (Name, Type) => A, t: TableType): A =
    v(TableIR.globalName, t.globalType)

  private def tableRow[A](v: (Name, Type) => A, t: TableType): A = v(TableIR.rowName, t.rowType)

  private def matrixGlobal[A](v: (Name, Type) => A, t: MatrixType): A =
    v(MatrixIR.globalName, t.globalType)

  private def sa[A](v: (Name, Type) => A, t: MatrixType): A = v(MatrixIR.colName, t.colType)
  private def va[A](v: (Name, Type) => A, t: MatrixType): A = v(MatrixIR.rowName, t.rowType)
  private def g[A](v: (Name, Type) => A, t: MatrixType): A = v(MatrixIR.entryName, t.entryType)

  // value nodes

  def block[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    bindings: IndexedSeq[Binding],
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] = {
    val bound = (0 until i).foldLeft(e)((e, k) => blockBind(e, bindings(k), children(k))(v))
    blockChild(bound, i, bindings)
  }

  private def blockChild[E[_]: Scopes, A](e: E[A], i: Int, bindings: IndexedSeq[Binding]): E[A] =
    if (i < bindings.length) Binds.blockBinding(e, bindings(i).scope) else e

  private def blockBind[E[_]: Scopes, A](e: E[A], b: Binding, value: BaseIR)(v: (Name, Type) => A)
    : E[A] =
    e.bind(b.scope, FastSeq(b.name -> v(b.name, typ(value))))

  def tailLoop[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    name: Name,
    params: IndexedSeq[Name],
    resultType: Type,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == params.length) {
      val argTypes = children.take(params.length).map(typ)
      Binds.tailLoop(
        e,
        name,
        params,
        params.lazyZip(argTypes).map(v),
        v(name, TTuple(TTuple(argTypes: _*), resultType)),
      )
    } else e

  def streamMap[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    name: Name,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) Binds.streamMap(e, name, v(name, eltType(children(0)))) else e

  def streamZip[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    names: IndexedSeq[Name],
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == names.length)
      Binds.streamZip(e, names, names.lazyZip(children).map((n, a) => v(n, eltType(a))))
    else e

  // nStreams is the number of streams joined
  def streamZipJoin[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    nStreams: Int,
    key: IndexedSeq[String],
    curKey: Name,
    curVals: Name,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == nStreams) {
      val struct = tcoerce[TStruct](eltType(children(0)))
      Binds.streamZipJoin(
        e,
        curKey,
        curVals,
        v(curKey, struct.typeAfterSelectNames(key)),
        v(curVals, TArray(struct)),
      )
    } else e

  def streamZipJoinProducers[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    ctxName: Name,
    key: IndexedSeq[String],
    curKey: Name,
    curVals: Name,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1)
      Binds.streamZipJoinProducersMakeProducer(e, ctxName, v(ctxName, eltType(children(0))))
    else if (i == 2) {
      val struct = tcoerce[TStruct](eltType(children(1)))
      Binds.streamZipJoinProducersJoinF(
        e,
        curKey,
        curVals,
        v(curKey, struct.typeAfterSelectNames(key)),
        v(curVals, TArray(struct)),
      )
    } else e

  def streamLeftIntervalJoin[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    lEltName: Name,
    rEltName: Name,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 2)
      Binds.streamLeftIntervalJoin(
        e,
        lEltName,
        rEltName,
        v(lEltName, eltType(children(0))),
        v(rEltName, TArray(eltType(children(1)))),
      )
    else e

  def streamFor[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    name: Name,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) Binds.streamFor(e, name, v(name, eltType(children(0)))) else e

  def streamFlatMap[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    name: Name,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) Binds.streamFlatMap(e, name, v(name, eltType(children(0)))) else e

  def streamFilter[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    name: Name,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) Binds.streamFilter(e, name, v(name, eltType(children(0)))) else e

  def streamTakeWhile[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    name: Name,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) Binds.streamTakeWhile(e, name, v(name, eltType(children(0)))) else e

  def streamDropWhile[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    name: Name,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) Binds.streamDropWhile(e, name, v(name, eltType(children(0)))) else e

  def streamFold[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    accumName: Name,
    valueName: Name,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 2)
      Binds.streamFold(
        e,
        accumName,
        valueName,
        v(accumName, typ(children(1))),
        v(valueName, eltType(children(0))),
      )
    else e

  def streamFold2[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    accumNames: IndexedSeq[Name],
    valueName: Name,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] = {
    val n = accumNames.length
    def accs: IndexedSeq[A] =
      accumNames.lazyZip(children.slice(1, n + 1)).map((a, c) => v(a, typ(c)))
    if (i < n + 1)
      e
    else if (i < 2 * n + 1)
      Binds.streamFold2Seq(e, valueName, accumNames, v(valueName, eltType(children(0))), accs)
    else
      Binds.streamFold2Result(e, accumNames, accs)
  }

  def streamBufferedAggregate[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    name: Name,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i >= 2) Binds.streamBufferedAggregate(e, name, v(name, eltType(children(0)))) else e

  def runAggScan[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    name: Name,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 2 || i == 3) Binds.runAggScan(e, name, v(name, eltType(children(0)))) else e

  def streamAgg[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    name: Name,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) Binds.streamAgg(e, name, v(name, eltType(children(0)))) else e

  def streamScan[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    accumName: Name,
    valueName: Name,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 2)
      Binds.streamScan(
        e,
        accumName,
        valueName,
        v(accumName, typ(children(1))),
        v(valueName, eltType(children(0))),
      )
    else e

  def streamAggScan[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    name: Name,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) Binds.streamAggScan(e, name, v(name, eltType(children(0)))) else e

  def streamJoinRightDistinct[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    l: Name,
    r: Name,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 2)
      Binds.streamJoinRightDistinct(
        e,
        l,
        r,
        v(l, eltType(children(0))),
        v(r, eltType(children(1))),
      )
    else e

  def arraySort[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    left: Name,
    right: Name,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) {
      val elt = eltType(children(0))
      Binds.arraySort(e, left, right, v(left, elt), v(right, elt))
    } else e

  // tieBreaker's left and right names, if it has one
  def arrayMaximalIndependentSet[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    tieBreaker: Option[(Name, Name)],
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    tieBreaker match {
      case Some((left, right)) if i == 1 =>
        val tupleType = TTuple(tcoerce[TBaseStruct](eltType(children(0))).types.head)
        Binds.arrayMaximalIndependentSet(e, left, right, v(left, tupleType), v(right, tupleType))
      case _ =>
        e
    }

  def aggArrayPerElement[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    elementName: Name,
    indexName: Name,
    isScan: Boolean,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] = {
    val s = aggScope(isScan)
    if (i == 0)
      Binds.aggSeq(e, s)
    else if (i == 1)
      Binds.aggArrayPerElement(
        e,
        s,
        elementName,
        indexName,
        v(elementName, eltType(children(0))),
        v(indexName, TInt32),
      )
    else
      Binds.aggInit(e, s)
  }

  def aggFold[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    accumName: Name,
    otherAccumName: Name,
    isScan: Boolean,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] = {
    val s = aggScope(isScan)
    if (i == 0)
      Binds.aggInit(e, s)
    else if (i == 1)
      Binds.aggFoldSeqOp(e, s, accumName, v(accumName, typ(children(0))))
    else
      Binds.aggFoldCombOp(
        e,
        accumName,
        otherAccumName,
        v(accumName, typ(children(0))),
        v(otherAccumName, typ(children(0))),
      )
  }

  def ndArrayMap[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    name: Name,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1)
      Binds.ndArrayMap(e, name, v(name, tcoerce[TNDArray](typ(children(0))).elementType))
    else e

  def ndArrayMap2[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    lName: Name,
    rName: Name,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 2)
      Binds.ndArrayMap2(
        e,
        lName,
        rName,
        v(lName, tcoerce[TNDArray](typ(children(0))).elementType),
        v(rName, tcoerce[TNDArray](typ(children(1))).elementType),
      )
    else e

  def collectDistributedArray[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    cname: Name,
    gname: Name,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 2)
      Binds.collectDistributedArray(
        e,
        cname,
        gname,
        v(cname, eltType(children(0))),
        v(gname, typ(children(1))),
      )
    else e

  def tableAggregate[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 0) e.fresh
    else {
      val t = tableType(children(0))
      Binds.tableAggregate(e, tableGlobal(v, t), tableRow(v, t))
    }

  def matrixAggregate[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 0) e.fresh
    else {
      val t = matrixType(children(0))
      Binds.matrixAggregate(e, matrixGlobal(v, t), sa(v, t), va(v, t), g(v, t))
    }

  // nInit is the number of init args
  def applyAggOp[E[_]: Scopes, A](e: E[A], i: Int, nInit: Int): E[A] =
    if (i < nInit) Binds.aggInit(e, AGG) else Binds.aggSeq(e, AGG)

  def applyScanOp[E[_]: Scopes, A](e: E[A], i: Int, nInit: Int): E[A] =
    if (i < nInit) Binds.aggInit(e, SCAN) else Binds.aggSeq(e, SCAN)

  def aggFilter[E[_]: Scopes, A](e: E[A], i: Int, isScan: Boolean): E[A] =
    if (i == 0) Binds.aggSeq(e, aggScope(isScan)) else e

  def aggGroupBy[E[_]: Scopes, A](e: E[A], i: Int, isScan: Boolean): E[A] =
    if (i == 0) Binds.aggSeq(e, aggScope(isScan)) else e

  def aggExplode[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    name: Name,
    isScan: Boolean,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] = {
    val s = aggScope(isScan)
    if (i == 0) Binds.aggSeq(e, s)
    else Binds.aggExplode(e, s, name, v(name, eltType(children(0))))
  }

  def relationalLet[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    name: Name,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 0) e.fresh else Binds.relationalLet(e, name, v(name, typ(children(0))))

  // table nodes

  def tableFilter[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) {
      val t = tableType(children(0))
      Binds.tableFilter(e, tableGlobal(v, t), tableRow(v, t))
    } else e.fresh

  def tableGen[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    cname: Name,
    gname: Name,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 2)
      Binds.tableGen(e, cname, gname, v(cname, eltType(children(0))), v(gname, typ(children(1))))
    else e.fresh

  def tableMapGlobals[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) Binds.tableMapGlobals(e, tableGlobal(v, tableType(children(0)))) else e.fresh

  def tableMapRows[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) {
      val t = tableType(children(0))
      Binds.tableMapRows(e, tableGlobal(v, t), tableRow(v, t))
    } else e.fresh

  def tableAggregateByKey[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) {
      val t = tableType(children(0))
      Binds.tableAggregateByKey(e, tableGlobal(v, t), tableRow(v, t))
    } else e.fresh

  def tableKeyByAndAggregate[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) {
      val t = tableType(children(0))
      Binds.tableKeyByAndAggregateExpr(e, tableGlobal(v, t), tableRow(v, t))
    } else if (i == 2) {
      val t = tableType(children(0))
      Binds.tableKeyByAndAggregateNewKey(e, tableGlobal(v, t), tableRow(v, t))
    } else e.fresh

  def tableMapPartitions[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    globalName: Name,
    partitionStreamName: Name,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) {
      val t = tableType(children(0))
      Binds.tableMapPartitions(
        e,
        globalName,
        partitionStreamName,
        v(globalName, t.globalType),
        v(partitionStreamName, TStream(t.rowType)),
      )
    } else e.fresh

  def relationalLetTable[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    name: Name,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) Binds.relationalLetTable(e, name, v(name, typ(children(0)))) else e.fresh

  // matrix nodes

  def matrixMapRows[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) {
      val t = matrixType(children(0))
      Binds.matrixMapRows(
        e,
        matrixGlobal(v, t),
        sa(v, t),
        va(v, t),
        g(v, t),
        v(Name("n_cols"), TInt32),
      )
    } else e.fresh

  def matrixFilterRows[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) {
      val t = matrixType(children(0))
      Binds.matrixFilterRows(e, matrixGlobal(v, t), va(v, t))
    } else e.fresh

  def matrixMapCols[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) {
      val t = matrixType(children(0))
      Binds.matrixMapCols(
        e,
        matrixGlobal(v, t),
        sa(v, t),
        va(v, t),
        g(v, t),
        v(Name("n_rows"), TInt64),
      )
    } else e.fresh

  def matrixFilterCols[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) {
      val t = matrixType(children(0))
      Binds.matrixFilterCols(e, matrixGlobal(v, t), sa(v, t))
    } else e.fresh

  def matrixMapEntries[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) {
      val t = matrixType(children(0))
      Binds.matrixMapEntries(e, matrixGlobal(v, t), sa(v, t), va(v, t), g(v, t))
    } else e.fresh

  def matrixFilterEntries[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) {
      val t = matrixType(children(0))
      Binds.matrixFilterEntries(e, matrixGlobal(v, t), sa(v, t), va(v, t), g(v, t))
    } else e.fresh

  def matrixMapGlobals[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) Binds.matrixMapGlobals(e, matrixGlobal(v, matrixType(children(0)))) else e.fresh

  def matrixAggregateColsByKey[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) {
      val t = matrixType(children(0))
      Binds.matrixAggregateColsByKeyEntryExpr(e, matrixGlobal(v, t), sa(v, t), va(v, t), g(v, t))
    } else if (i == 2) {
      val t = matrixType(children(0))
      Binds.matrixAggregateColsByKeyColExpr(e, matrixGlobal(v, t), sa(v, t))
    } else e.fresh

  def matrixAggregateRowsByKey[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) {
      val t = matrixType(children(0))
      Binds.matrixAggregateRowsByKeyEntryExpr(e, matrixGlobal(v, t), sa(v, t), va(v, t), g(v, t))
    } else if (i == 2) {
      val t = matrixType(children(0))
      Binds.matrixAggregateRowsByKeyRowExpr(e, matrixGlobal(v, t), va(v, t))
    } else e.fresh

  def relationalLetMatrixTable[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    name: Name,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) Binds.relationalLetMatrixTable(e, name, v(name, typ(children(0)))) else e.fresh

  // block matrix nodes

  def blockMatrixMap[E[_]: Scopes, A](e: E[A], i: Int, eltName: Name)(v: (Name, Type) => A): E[A] =
    if (i == 1) Binds.blockMatrixMap(e, eltName, v(eltName, TFloat64)) else e.fresh

  def blockMatrixMap2[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    leftName: Name,
    rightName: Name,
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 2)
      Binds.blockMatrixMap2(e, leftName, rightName, v(leftName, TFloat64), v(rightName, TFloat64))
    else e.fresh
}

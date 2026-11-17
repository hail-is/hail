package is.hail.expr.ir

import is.hail.collection.FastSeq
import is.hail.expr.ir.Scope.{AGG, EVAL, SCAN}
import is.hail.expr.ir.Scopes.ScopesSyntax
import is.hail.expr.ir.defs._
import is.hail.types.tcoerce
import is.hail.types.virtual._
import is.hail.types.virtual.TIterable.elementType
import is.hail.utils.StackSafe._

import scala.annotation.switch

object ChildEnv {

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
    (ir.tag.family.toInt: @switch) match {
      case Family.Core => getCore(e, ir, cs, i)(v)
      case Family.Stream => getStream(e, ir, cs, i)(v)
      case Family.Collection => getCollection(e, ir, cs, i)(v)
      case Family.Agg => getAgg(e, ir, cs, i)(v)
      case Family.NDArray => getNDArray(e, ir, cs, i)(v)
      case Family.Relational => getRelational(e, ir, cs, i)(v)
      case Family.Table => getTable(e, ir, cs, i)(v)
      case Family.Matrix => getMatrix(e, ir, cs, i)(v)
      case Family.BlockMatrix => getBlockMatrix(e, ir, i)(v)
      case _ => e
    }

  private def getCore[E[_]: Scopes, A](
    e: E[A],
    ir: BaseIR,
    cs: IndexedSeq[BaseIR],
    i: Int,
  )(
    v: (Name, Type) => A
  ): E[A] =
    (ir.tag.toInt: @switch) match {
      case Tag.Block =>
        block(e, i, ir.asInstanceOf[Block].bindings)(v)
      case Tag.TailLoop =>
        val x = ir.asInstanceOf[TailLoop]
        tailLoop(e, i, x.name, x.params.map(_._1), x.resultType, cs)(v)
      case _ =>
        e
    }

  private def getStream[E[_]: Scopes, A](
    e: E[A],
    ir: BaseIR,
    cs: IndexedSeq[BaseIR],
    i: Int,
  )(
    v: (Name, Type) => A
  ): E[A] =
    (ir.tag.toInt: @switch) match {
      case Tag.StreamBufferedAggregate =>
        streamBufferedAggregate(e, i, ir.asInstanceOf[StreamBufferedAggregate].name, cs)(v)
      case Tag.StreamMap =>
        streamMap(e, i, ir.asInstanceOf[StreamMap].name, cs)(v)
      case Tag.StreamTakeWhile =>
        streamTakeWhile(e, i, ir.asInstanceOf[StreamTakeWhile].elementName, cs)(v)
      case Tag.StreamDropWhile =>
        streamDropWhile(e, i, ir.asInstanceOf[StreamDropWhile].elementName, cs)(v)
      case Tag.StreamZip =>
        streamZip(e, i, ir.asInstanceOf[StreamZip].names, cs)(v)
      case Tag.StreamZipJoinProducers =>
        val x = ir.asInstanceOf[StreamZipJoinProducers]
        streamZipJoinProducers(e, i, x.ctxName, x.key, x.curKey, x.curVals, cs)(v)
      case Tag.StreamZipJoin =>
        val x = ir.asInstanceOf[StreamZipJoin]
        streamZipJoin(e, i, x.as.length, x.key, x.curKey, x.curVals, cs)(v)
      case Tag.StreamFilter =>
        streamFilter(e, i, ir.asInstanceOf[StreamFilter].name, cs)(v)
      case Tag.StreamFlatMap =>
        streamFlatMap(e, i, ir.asInstanceOf[StreamFlatMap].name, cs)(v)
      case Tag.StreamFold =>
        val x = ir.asInstanceOf[StreamFold]
        streamFold(e, i, x.accumName, x.valueName, cs)(v)
      case Tag.StreamFold2 =>
        val x = ir.asInstanceOf[StreamFold2]
        streamFold2(e, i, x.accum.map(_._1), x.valueName, cs)(v)
      case Tag.StreamScan =>
        val x = ir.asInstanceOf[StreamScan]
        streamScan(e, i, x.accumName, x.valueName, cs)(v)
      case Tag.StreamFor =>
        streamFor(e, i, ir.asInstanceOf[StreamFor].valueName, cs)(v)
      case Tag.StreamAgg =>
        streamAgg(e, i, ir.asInstanceOf[StreamAgg].name, cs)(v)
      case Tag.StreamAggScan =>
        streamAggScan(e, i, ir.asInstanceOf[StreamAggScan].name, cs)(v)
      case Tag.StreamLeftIntervalJoin =>
        val x = ir.asInstanceOf[StreamLeftIntervalJoin]
        streamLeftIntervalJoin(e, i, x.lname, x.rname, cs)(v)
      case Tag.StreamJoinRightDistinct =>
        val x = ir.asInstanceOf[StreamJoinRightDistinct]
        streamJoinRightDistinct(e, i, x.l, x.r, cs)(v)
      case _ =>
        e
    }

  private def getCollection[E[_]: Scopes, A](
    e: E[A],
    ir: BaseIR,
    cs: IndexedSeq[BaseIR],
    i: Int,
  )(
    v: (Name, Type) => A
  ): E[A] =
    (ir.tag.toInt: @switch) match {
      case Tag.ArrayMaximalIndependentSet =>
        val x = ir.asInstanceOf[ArrayMaximalIndependentSet]
        arrayMaximalIndependentSet(e, i, x.tieBreaker.map { case (l, r, _) => (l, r) }, cs)(v)
      case Tag.ArraySort =>
        val x = ir.asInstanceOf[ArraySort]
        arraySort(e, i, x.left, x.right, cs)(v)
      case Tag.CollectDistributedArray =>
        val x = ir.asInstanceOf[CollectDistributedArray]
        collectDistributedArray(e, i, x.cname, x.gname, cs)(v)
      case _ =>
        e
    }

  private def getAgg[E[_]: Scopes, A](
    e: E[A],
    ir: BaseIR,
    cs: IndexedSeq[BaseIR],
    i: Int,
  )(
    v: (Name, Type) => A
  ): E[A] =
    (ir.tag.toInt: @switch) match {
      case Tag.AggFilter =>
        aggFilter(e, i, ir.asInstanceOf[AggFilter].isScan)
      case Tag.AggExplode =>
        val x = ir.asInstanceOf[AggExplode]
        aggExplode(e, i, x.name, x.isScan, cs)(v)
      case Tag.AggGroupBy =>
        aggGroupBy(e, i, ir.asInstanceOf[AggGroupBy].isScan)
      case Tag.AggArrayPerElement =>
        val x = ir.asInstanceOf[AggArrayPerElement]
        aggArrayPerElement(e, i, x.elementName, x.indexName, x.isScan, cs)(v)
      case Tag.AggFold =>
        val x = ir.asInstanceOf[AggFold]
        aggFold(e, i, x.accumName, x.otherAccumName, x.isScan, cs)(v)
      case Tag.ApplyAggOp =>
        applyAggOp(e, i, ir.asInstanceOf[ApplyAggOp].initOpArgs.length)
      case Tag.ApplyScanOp =>
        applyScanOp(e, i, ir.asInstanceOf[ApplyScanOp].initOpArgs.length)
      case Tag.RunAggScan =>
        runAggScan(e, i, ir.asInstanceOf[RunAggScan].name, cs)(v)
      case _ =>
        e
    }

  private def getNDArray[E[_]: Scopes, A](
    e: E[A],
    ir: BaseIR,
    cs: IndexedSeq[BaseIR],
    i: Int,
  )(
    v: (Name, Type) => A
  ): E[A] =
    (ir.tag.toInt: @switch) match {
      case Tag.NDArrayMap =>
        ndArrayMap(e, i, ir.asInstanceOf[NDArrayMap].valueName, cs)(v)
      case Tag.NDArrayMap2 =>
        val x = ir.asInstanceOf[NDArrayMap2]
        ndArrayMap2(e, i, x.lName, x.rName, cs)(v)
      case _ => e
    }

  private def getRelational[E[_]: Scopes, A](
    e: E[A],
    ir: BaseIR,
    cs: IndexedSeq[BaseIR],
    i: Int,
  )(
    v: (Name, Type) => A
  ): E[A] =
    (ir.tag.toInt: @switch) match {
      case Tag.TableAggregate => tableAggregate(e, i, cs)(v)
      case _ => e
    }

  private def getTable[E[_]: Scopes, A](
    e: E[A],
    ir: BaseIR,
    cs: IndexedSeq[BaseIR],
    i: Int,
  )(
    v: (Name, Type) => A
  ): E[A] =
    (ir.tag.toInt: @switch) match {
      case Tag.TableGen =>
        val x = ir.asInstanceOf[TableGen]
        tableGen(e, i, x.cname, x.gname, cs)(v)
      case Tag.TableMapPartitions =>
        val x = ir.asInstanceOf[TableMapPartitions]
        tableMapPartitions(e, i, x.globalName, x.partitionStreamName, cs)(v)
      case Tag.TableMapRows =>
        tableMapRows(e, i, cs)(v)
      case Tag.TableKeyByAndAggregate =>
        tableKeyByAndAggregate(e, i, cs)(v)
      case Tag.TableAggregateByKey =>
        tableAggregateByKey(e, i, cs)(v)
      case Tag.TableFilter =>
        tableFilter(e, i, cs)(v)
      case Tag.TableMapGlobals =>
        tableMapGlobals(e, i, cs)(v)
      case Tag.TableMultiWrite | Tag.TableWrite =>
        e
      case _ =>
        e.empty
    }

  private def getMatrix[E[_]: Scopes, A](
    e: E[A],
    ir: BaseIR,
    cs: IndexedSeq[BaseIR],
    i: Int,
  )(
    v: (Name, Type) => A
  ): E[A] =
    (ir.tag.toInt: @switch) match {
      case Tag.MatrixAggregate =>
        matrixAggregate(e, i, cs)(v)
      case Tag.MatrixAggregateRowsByKey =>
        matrixAggregateRowsByKey(e, i, cs)(v)
      case Tag.MatrixAggregateColsByKey =>
        matrixAggregateColsByKey(e, i, cs)(v)
      case Tag.MatrixFilterCols =>
        matrixFilterCols(e, i, cs)(v)
      case Tag.MatrixFilterRows =>
        matrixFilterRows(e, i, cs)(v)
      case Tag.MatrixMapEntries =>
        matrixMapEntries(e, i, cs)(v)
      case Tag.MatrixMapRows =>
        matrixMapRows(e, i, cs)(v)
      case Tag.MatrixMapCols =>
        matrixMapCols(e, i, cs)(v)
      case Tag.MatrixMapGlobals =>
        matrixMapGlobals(e, i, cs)(v)
      case Tag.MatrixFilterEntries =>
        matrixFilterEntries(e, i, cs)(v)
      case Tag.MatrixMultiWrite |
          Tag.MatrixCount |
          Tag.MatrixWrite |
          Tag.MatrixToValueApply =>
        e
      case _ =>
        e.empty
    }

  private def getBlockMatrix[E[_]: Scopes, A](
    e: E[A],
    ir: BaseIR,
    i: Int,
  )(
    v: (Name, Type) => A
  ): E[A] =
    (ir.tag.toInt: @switch) match {
      case Tag.BlockMatrixMap =>
        blockMatrixMap(e, i, ir.asInstanceOf[BlockMatrixMap].eltName)(v)
      case Tag.BlockMatrixMap2 =>
        val x = ir.asInstanceOf[BlockMatrixMap2]
        blockMatrixMap2(e, i, x.leftName, x.rightName)(v)
      case Tag.BlockMatrixWrite | Tag.BlockMatrixMultiWrite =>
        e
      case _ =>
        e.empty
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
  )(
    v: (Name, Type) => A
  ): E[A] = {
    val s = if (i < bindings.length) bindings(i).scope else EVAL
    (0 until i).foldLeft(Binds.aggSeq(e, s)) { (e, k) =>
      val Binding(name, expr, scope) = bindings(k)
      if (s == EVAL) e.bind(scope, FastSeq(name -> v(name, expr.typ)))
      else if (scope == s) e.bind(EVAL, FastSeq(name -> v(name, expr.typ)))
      else e
    }
  }

  /** Walks Block `b`'s children (each binding's value, then the body), threading `b`'s env so the
    * walk is linear in its bindings. `f` takes child `i`, `i` and child `i`'s env, and returns
    * child `i`'s final form, whose type binding `i` binds.
    */
  def walkBlock[E[_]: Scopes, A](
    b: Block,
    e: E[A],
  )(
    v: (Name, Type) => A
  )(
    f: (BaseIR, Int, E[A]) => BaseIR
  ): Unit = {
    var env = e
    for (i <- b.bindings.indices) {
      val Binding(name, value, scope) = b.bindings(i)
      val c = f(value, i, Binds.aggSeq(env, scope))
      env = env.bind(scope, FastSeq(name -> v(name, typ(c))))
    }
    f(b.body, b.bindings.length, env): Unit
  }

  def walkBlockStackSafe[E[_]: Scopes, A](
    b: Block,
    e: E[A],
  )(
    v: (Name, Type) => A
  )(
    f: (BaseIR, Int, E[A]) => StackFrame[BaseIR]
  ): StackFrame[Unit] = {
    var env = e
    b.bindings.indices.foreachRecur { i =>
      val Binding(name, value, scope) = b.bindings(i)
      f(value, i, Binds.aggSeq(env, scope)).map { c =>
        env = env.bind(scope, FastSeq(name -> v(name, typ(c))))
      }
    }.flatMap(_ => f(b.body, b.bindings.length, env).map(_ => ()))
  }

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
    if (i == 1) Binds.streamMap(e, name, v(name, eltType(children(0))))
    else e

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
      val struct = eltType(children(0)).asInstanceOf[TStruct]
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
    if (i == 1) Binds.streamFor(e, name, v(name, eltType(children(0))))
    else e

  def streamFlatMap[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    name: Name,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) Binds.streamFlatMap(e, name, v(name, eltType(children(0))))
    else e

  def streamFilter[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    name: Name,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) Binds.streamFilter(e, name, v(name, eltType(children(0))))
    else e

  def streamTakeWhile[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    name: Name,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) Binds.streamTakeWhile(e, name, v(name, eltType(children(0))))
    else e

  def streamDropWhile[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    name: Name,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) Binds.streamDropWhile(e, name, v(name, eltType(children(0))))
    else e

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
    if (i >= 2) Binds.streamBufferedAggregate(e, name, v(name, eltType(children(0))))
    else e

  def runAggScan[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    name: Name,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 2 || i == 3) Binds.runAggScan(e, name, v(name, eltType(children(0))))
    else e

  def streamAgg[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    name: Name,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) Binds.streamAgg(e, name, v(name, eltType(children(0))))
    else e

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
    if (i == 1) Binds.streamAggScan(e, name, v(name, eltType(children(0))))
    else e

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
    if (i == 0) e.empty
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
    if (i == 0) e.empty
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
    } else e.empty

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
    else e.empty

  def tableMapGlobals[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) {
      val t = tableType(children(0))
      Binds.tableMapGlobals(e, tableGlobal(v, t), tableRow(v, t))
    } else e.empty

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
    } else e.empty

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
    } else e.empty

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
    } else e.empty

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
    } else e.empty

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
    } else e.empty

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
    } else e.empty

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
    } else e.empty

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
    } else e.empty

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
    } else e.empty

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
    } else e.empty

  def matrixMapGlobals[E[_]: Scopes, A](
    e: E[A],
    i: Int,
    children: IndexedSeq[BaseIR],
  )(
    v: (Name, Type) => A
  ): E[A] =
    if (i == 1) Binds.matrixMapGlobals(e, matrixGlobal(v, matrixType(children(0)))) else e.empty

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
    } else e.empty

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
    } else e.empty

  // block matrix nodes

  def blockMatrixMap[E[_]: Scopes, A](e: E[A], i: Int, eltName: Name)(v: (Name, Type) => A): E[A] =
    if (i == 1) Binds.blockMatrixMap(e, eltName, v(eltName, TFloat64)) else e.empty

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
    else e.empty
}

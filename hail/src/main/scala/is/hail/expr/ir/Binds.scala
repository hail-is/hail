package is.hail.expr.ir

import is.hail.types.tcoerce
import is.hail.types.virtual._
import is.hail.types.virtual.TIterable.elementType

object SegregatedBindingEnv {
  def apply[A, B](env: BindingEnv[A]): SegregatedBindingEnv[A, B] =
    SegregatedBindingEnv(env, env.dropBindings)
}

case class SegregatedBindingEnv[A, B](
  childEnvWithoutBindings: BindingEnv[A],
  newBindings: BindingEnv[B],
) extends GenericBindingEnv[SegregatedBindingEnv[A, B], B] {
  def unified(implicit ev: BindingEnv[B] =:= BindingEnv[A]): BindingEnv[A] =
    childEnvWithoutBindings.merge(newBindings)

  def mapNewBindings[C](f: (String, B) => C): SegregatedBindingEnv[A, C] = SegregatedBindingEnv(
    childEnvWithoutBindings,
    newBindings.mapValuesWithKey(f),
  )

  override def promoteAgg: SegregatedBindingEnv[A, B] = SegregatedBindingEnv(
    childEnvWithoutBindings.promoteAgg,
    newBindings.promoteAgg,
  )

  override def promoteScan: SegregatedBindingEnv[A, B] = SegregatedBindingEnv(
    childEnvWithoutBindings.promoteScan,
    newBindings.promoteScan,
  )

  override def bindEval(bindings: (String, B)*): SegregatedBindingEnv[A, B] =
    copy(newBindings = newBindings.bindEval(bindings: _*))

  override def dropEval: SegregatedBindingEnv[A, B] = SegregatedBindingEnv(
    childEnvWithoutBindings.copy(eval = Env.empty),
    newBindings.copy(eval = Env.empty),
  )

  override def bindAgg(bindings: (String, B)*): SegregatedBindingEnv[A, B] =
    copy(newBindings = newBindings.bindAgg(bindings: _*))

  override def bindScan(bindings: (String, B)*): SegregatedBindingEnv[A, B] =
    copy(newBindings = newBindings.bindScan(bindings: _*))

  override def createAgg: SegregatedBindingEnv[A, B] = SegregatedBindingEnv(
    childEnvWithoutBindings.createAgg,
    newBindings.createAgg,
  )

  override def createScan: SegregatedBindingEnv[A, B] = SegregatedBindingEnv(
    childEnvWithoutBindings.createScan,
    newBindings.createScan,
  )

  override def noAgg: SegregatedBindingEnv[A, B] = SegregatedBindingEnv(
    childEnvWithoutBindings.noAgg,
    newBindings.noAgg,
  )

  override def noScan: SegregatedBindingEnv[A, B] = SegregatedBindingEnv(
    childEnvWithoutBindings.noScan,
    newBindings.noScan,
  )

  override def onlyRelational(keepAggCapabilities: Boolean = false): SegregatedBindingEnv[A, B] =
    SegregatedBindingEnv(
      childEnvWithoutBindings.onlyRelational(keepAggCapabilities),
      newBindings.onlyRelational(keepAggCapabilities),
    )

  override def bindRelational(bindings: (String, B)*): SegregatedBindingEnv[A, B] =
    copy(newBindings = newBindings.bindRelational(bindings: _*))
}

case class EvalOnlyBindingEnv[T](env: Env[T]) extends GenericBindingEnv[EvalOnlyBindingEnv[T], T] {
  override def promoteAgg: EvalOnlyBindingEnv[T] =
    EvalOnlyBindingEnv(Env.empty)

  override def promoteScan: EvalOnlyBindingEnv[T] =
    EvalOnlyBindingEnv(Env.empty)

  override def bindEval(bindings: (String, T)*): EvalOnlyBindingEnv[T] =
    EvalOnlyBindingEnv(env.bindIterable(bindings))

  override def dropEval: EvalOnlyBindingEnv[T] =
    EvalOnlyBindingEnv(Env.empty)

  override def bindAgg(bindings: (String, T)*): EvalOnlyBindingEnv[T] =
    this

  override def bindScan(bindings: (String, T)*): EvalOnlyBindingEnv[T] =
    this

  override def createAgg: EvalOnlyBindingEnv[T] =
    this

  override def createScan: EvalOnlyBindingEnv[T] =
    this

  override def noAgg: EvalOnlyBindingEnv[T] =
    this

  override def noScan: EvalOnlyBindingEnv[T] =
    this

  override def onlyRelational(keepAggCapabilities: Boolean = false): EvalOnlyBindingEnv[T] =
    EvalOnlyBindingEnv(Env.empty)

  override def bindRelational(bindings: (String, T)*): EvalOnlyBindingEnv[T] =
    this
}

object Binds {
  def apply(x: IR, v: String, i: Int): Boolean =
    Bindings(x, i, EvalOnlyBindingEnv(Env.empty[Type])).env.contains(v)
}

object Bindings {

  /** Returns the environment of the `i`th child of `ir` given the environment of the parent node
    * `ir`.
    */
  def apply[E <: GenericBindingEnv[E, Type]](ir: BaseIR, i: Int, baseEnv: E): E =
    ir match {
      case ir: MatrixIR => childEnvMatrix(ir, i, baseEnv)
      case ir: TableIR => childEnvTable(ir, i, baseEnv)
      case ir: BlockMatrixIR => childEnvBlockMatrix(ir, i, baseEnv)
      case ir: IR => childEnvValue(ir, i, baseEnv)
    }

  /** Like [[Bindings.apply]], but keeps separate any new bindings introduced by `ir`. Always
    * satisfies the identity
    * {{{
    * Bindings.segregated(ir, i, baseEnv).unified == Bindings(ir, i, baseEnv)
    * }}}
    */
  def segregated[A](ir: BaseIR, i: Int, baseEnv: BindingEnv[A]): SegregatedBindingEnv[A, Type] =
    apply(ir, i, SegregatedBindingEnv(baseEnv))

  private def childEnvMatrix[E <: GenericBindingEnv[E, Type]](ir: MatrixIR, i: Int, _baseEnv: E)
    : E = {
    val baseEnv = _baseEnv.onlyRelational()
    ir match {
      case MatrixMapRows(child, _) if i == 1 =>
        baseEnv
          .createAgg.createScan
          .bindEval(child.typ.rowBindings: _*)
          .bindEval("n_cols" -> TInt32)
          .bindAgg(child.typ.entryBindings: _*)
          .bindScan(child.typ.rowBindings: _*)
      case MatrixFilterRows(child, _) if i == 1 =>
        baseEnv.bindEval(child.typ.rowBindings: _*)
      case MatrixMapCols(child, _, _) if i == 1 =>
        baseEnv
          .createAgg.createScan
          .bindEval(child.typ.colBindings: _*)
          .bindEval("n_rows" -> TInt64)
          .bindAgg(child.typ.entryBindings: _*)
          .bindScan(child.typ.colBindings: _*)
      case MatrixFilterCols(child, _) if i == 1 =>
        baseEnv.bindEval(child.typ.colBindings: _*)
      case MatrixMapEntries(child, _) if i == 1 =>
        baseEnv.bindEval(child.typ.entryBindings: _*)
      case MatrixFilterEntries(child, _) if i == 1 =>
        baseEnv.bindEval(child.typ.entryBindings: _*)
      case MatrixMapGlobals(child, _) if i == 1 =>
        baseEnv.bindEval(child.typ.globalBindings: _*)
      case MatrixAggregateColsByKey(child, _, _) =>
        if (i == 1)
          baseEnv
            .bindEval(child.typ.rowBindings: _*)
            .createAgg.bindAgg(child.typ.entryBindings: _*)
        else if (i == 2)
          baseEnv
            .bindEval(child.typ.globalBindings: _*)
            .createAgg.bindAgg(child.typ.colBindings: _*)
        else baseEnv
      case MatrixAggregateRowsByKey(child, _, _) =>
        if (i == 1)
          baseEnv
            .bindEval(child.typ.colBindings: _*)
            .createAgg.bindAgg(child.typ.entryBindings: _*)
        else if (i == 2)
          baseEnv
            .bindEval(child.typ.globalBindings: _*)
            .createAgg.bindAgg(child.typ.rowBindings: _*)
        else baseEnv
      case RelationalLetMatrixTable(name, value, _) if i == 1 =>
        baseEnv.bindRelational(name -> value.typ)
      case _ =>
        baseEnv
    }
  }

  private def childEnvTable[E <: GenericBindingEnv[E, Type]](ir: TableIR, i: Int, _baseEnv: E)
    : E = {
    val baseEnv = _baseEnv.onlyRelational()
    ir match {
      case TableFilter(child, _) if i == 1 =>
        baseEnv.bindEval(child.typ.rowBindings: _*)
      case TableGen(contexts, globals, cname, gname, _, _, _) if i == 2 =>
        baseEnv.bindEval(
          cname -> elementType(contexts.typ),
          gname -> globals.typ,
        )
      case TableMapGlobals(child, _) if i == 1 =>
        baseEnv.bindEval(child.typ.globalBindings: _*)
      case TableMapRows(child, _) if i == 1 =>
        baseEnv
          .bindEval(child.typ.rowBindings: _*)
          .createScan.bindScan(child.typ.rowBindings: _*)
      case TableAggregateByKey(child, _) if i == 1 =>
        baseEnv
          .bindEval(child.typ.globalBindings: _*)
          .createAgg.bindAgg(child.typ.rowBindings: _*)
      case TableKeyByAndAggregate(child, _, _, _, _) =>
        if (i == 1)
          baseEnv
            .bindEval(child.typ.globalBindings: _*)
            .createAgg.bindAgg(child.typ.rowBindings: _*)
        else if (i == 2)
          baseEnv.bindEval(child.typ.rowBindings: _*)
        else baseEnv
      case TableMapPartitions(child, g, p, _, _, _) if i == 1 =>
        baseEnv.bindEval(
          g -> child.typ.globalType,
          p -> TStream(child.typ.rowType),
        )
      case RelationalLetTable(name, value, _) if i == 1 =>
        baseEnv.bindRelational(name -> value.typ)
      case _ =>
        baseEnv
    }
  }

  private def childEnvBlockMatrix[E <: GenericBindingEnv[E, Type]](
    ir: BlockMatrixIR,
    i: Int,
    _baseEnv: E,
  ): E = {
    val baseEnv = _baseEnv.onlyRelational()
    ir match {
      case BlockMatrixMap(_, eltName, _, _) if i == 1 =>
        baseEnv.bindEval(eltName -> TFloat64)
      case BlockMatrixMap2(_, _, lName, rName, _, _) if i == 2 =>
        baseEnv.bindEval(lName -> TFloat64, rName -> TFloat64)
      case RelationalLetBlockMatrix(name, value, _) if i == 1 =>
        baseEnv.bindRelational(name -> value.typ)
      case _ =>
        baseEnv
    }
  }

  private def childEnvValue[E <: GenericBindingEnv[E, Type]](ir: IR, i: Int, baseEnv: E): E =
    ir match {
      case Let(bindings, _) =>
        val result = Array.ofDim[(String, Type)](i)
        for (k <- 0 until i) result(k) = bindings(k)._1 -> bindings(k)._2.typ
        baseEnv.bindEval(result: _*)
      case TailLoop(name, args, resultType, _) if i == args.length =>
        baseEnv
          .bindEval(args.map { case (name, ir) => name -> ir.typ }: _*)
          .bindEval(name -> TTuple(TTuple(args.map(_._2.typ): _*), resultType))
      case StreamMap(a, name, _) if i == 1 =>
        baseEnv.bindEval(name -> elementType(a.typ))
      case StreamZip(as, names, _, _, _) if i == as.length =>
        baseEnv.bindEval(names.zip(as.map(a => elementType(a.typ))): _*)
      case StreamZipJoin(as, key, curKey, curVals, _) if i == as.length =>
        val eltType = tcoerce[TStruct](elementType(as.head.typ))
        baseEnv.bindEval(
          curKey -> eltType.typeAfterSelectNames(key),
          curVals -> TArray(eltType),
        )
      case StreamZipJoinProducers(contexts, ctxName, makeProducer, key, curKey, curVals, _) =>
        if (i == 1) {
          val contextType = elementType(contexts.typ)
          baseEnv.bindEval(ctxName -> contextType)
        } else if (i == 2) {
          val eltType = tcoerce[TStruct](elementType(makeProducer.typ))
          baseEnv.bindEval(
            curKey -> eltType.typeAfterSelectNames(key),
            curVals -> TArray(eltType),
          )
        } else baseEnv
      case StreamLeftIntervalJoin(left, right, _, _, lEltName, rEltName, _) if i == 2 =>
        baseEnv.bindEval(
          lEltName -> elementType(left.typ),
          rEltName -> TArray(elementType(right.typ)),
        )
      case StreamFor(a, name, _) if i == 1 =>
        baseEnv.bindEval(name -> elementType(a.typ))
      case StreamFlatMap(a, name, _) if i == 1 =>
        baseEnv.bindEval(name -> elementType(a.typ))
      case StreamFilter(a, name, _) if i == 1 =>
        baseEnv.bindEval(name -> elementType(a.typ))
      case StreamTakeWhile(a, name, _) if i == 1 =>
        baseEnv.bindEval(name -> elementType(a.typ))
      case StreamDropWhile(a, name, _) if i == 1 =>
        baseEnv.bindEval(name -> elementType(a.typ))
      case StreamFold(a, zero, accumName, valueName, _) if i == 2 =>
        baseEnv.bindEval(accumName -> zero.typ, valueName -> elementType(a.typ))
      case StreamFold2(a, accum, valueName, _, _) =>
        if (i <= accum.length)
          baseEnv
        else if (i < 2 * accum.length + 1)
          baseEnv
            .bindEval(valueName -> elementType(a.typ))
            .bindEval(accum.map { case (name, value) => (name, value.typ) }: _*)
        else
          baseEnv.bindEval(accum.map { case (name, value) => (name, value.typ) }: _*)
      case StreamBufferedAggregate(stream, _, _, _, name, _, _) if i > 0 =>
        baseEnv.bindEval(name -> elementType(stream.typ))
      case RunAggScan(a, name, _, _, _, _) if i == 2 || i == 3 =>
        baseEnv.bindEval(name -> elementType(a.typ))
      case StreamScan(a, zero, accumName, valueName, _) if i == 2 =>
        baseEnv.bindEval(
          accumName -> zero.typ,
          valueName -> elementType(a.typ),
        )
      case StreamAggScan(a, name, _) if i == 1 =>
        val eltType = elementType(a.typ)
        baseEnv
          .bindEval(name -> eltType)
          .createScan.bindScan(name -> eltType)
      case StreamJoinRightDistinct(ll, rr, _, _, l, r, _, _) if i == 2 =>
        baseEnv.bindEval(
          l -> elementType(ll.typ),
          r -> elementType(rr.typ),
        )
      case ArraySort(a, left, right, _) if i == 1 =>
        baseEnv.bindEval(
          left -> elementType(a.typ),
          right -> elementType(a.typ),
        )
      case ArrayMaximalIndependentSet(a, Some((left, right, _))) if i == 1 =>
        val typ = tcoerce[TBaseStruct](elementType(a.typ)).types.head
        val tupleType = TTuple(typ)
        baseEnv.dropEval.bindEval(left -> tupleType, right -> tupleType)
      case AggArrayPerElement(a, elementName, indexName, _, _, isScan) =>
        if (i == 0) baseEnv.promoteAggOrScan(isScan)
        else if (i == 1)
          baseEnv
            .bindEval(indexName -> TInt32)
            .bindAggOrScan(
              isScan,
              elementName -> elementType(a.typ),
              indexName -> TInt32,
            )
        else baseEnv
      case AggFold(zero, _, _, accumName, otherAccumName, isScan) =>
        if (i == 0) baseEnv.noAggOrScan(isScan)
        else if (i == 1) baseEnv.promoteAggOrScan(isScan).bindEval(accumName -> zero.typ)
        else baseEnv.dropEval.noAggOrScan(isScan)
          .bindEval(accumName -> zero.typ, otherAccumName -> zero.typ)
      case NDArrayMap(nd, name, _) if i == 1 =>
        baseEnv.bindEval(name -> tcoerce[TNDArray](nd.typ).elementType)
      case NDArrayMap2(l, r, lName, rName, _, _) if i == 2 =>
        baseEnv.bindEval(
          lName -> tcoerce[TNDArray](l.typ).elementType,
          rName -> tcoerce[TNDArray](r.typ).elementType,
        )
      case CollectDistributedArray(contexts, globals, cname, gname, _, _, _, _) if i == 2 =>
        baseEnv.onlyRelational().bindEval(
          cname -> elementType(contexts.typ),
          gname -> globals.typ,
        )
      case TableAggregate(child, _) =>
        if (i == 1)
          baseEnv.onlyRelational()
            .bindEval(child.typ.globalBindings: _*)
            .createAgg.bindAgg(child.typ.rowBindings: _*)
        else baseEnv.onlyRelational()
      case MatrixAggregate(child, _) =>
        if (i == 1)
          baseEnv.onlyRelational()
            .bindEval(child.typ.globalBindings: _*)
            .createAgg.bindAgg(child.typ.entryBindings: _*)
        else baseEnv.onlyRelational()
      case ApplyAggOp(init, _, _) =>
        if (i < init.length) baseEnv.noAgg
        else baseEnv.promoteAgg
      case ApplyScanOp(init, _, _) =>
        if (i < init.length) baseEnv.noScan
        else baseEnv.promoteScan
      case AggLet(name, value, _, isScan) =>
        if (i == 0) baseEnv.promoteAggOrScan(isScan)
        else baseEnv.bindAggOrScan(isScan, name -> value.typ)
      case AggFilter(_, _, isScan) if i == 0 =>
        baseEnv.promoteAggOrScan(isScan)
      case AggGroupBy(_, _, isScan) if i == 0 =>
        baseEnv.promoteAggOrScan(isScan)
      case AggExplode(a, name, _, isScan) =>
        if (i == 0) baseEnv.promoteAggOrScan(isScan)
        else baseEnv.bindAggOrScan(isScan, name -> elementType(a.typ))
      case StreamAgg(a, name, _) if i == 1 =>
        baseEnv.createAgg
          .bindAgg(name -> elementType(a.typ))
      case RelationalLet(name, value, _) =>
        if (i == 1)
          baseEnv.noAgg.noScan.bindRelational(name -> value.typ)
        else
          baseEnv.onlyRelational()
      case _: LiftMeOut =>
        baseEnv.onlyRelational(keepAggCapabilities = true)
      case _ =>
        if (UsesAggEnv(ir, i)) baseEnv.promoteAgg
        else if (UsesScanEnv(ir, i)) baseEnv.promoteScan
        else baseEnv
    }
}

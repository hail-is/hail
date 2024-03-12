package is.hail.expr.ir

import is.hail.types.tcoerce
import is.hail.types.virtual._
import is.hail.types.virtual.TIterable.elementType
import is.hail.utils._

object Binds {
  def apply(x: IR, v: String, i: Int): Boolean = Bindings(x, i).exists(_._1 == v)
}

object Bindings {
  private val empty: Array[(String, Type)] = Array()

  // A call to Bindings(x, i) may only query the types of children with
  // index < i
  def apply(x: BaseIR, i: Int): Iterable[(String, Type)] = x match {
    case Let(bindings, _) =>
      val result = Array.ofDim[(String, Type)](i)
      for (k <- 0 until i) result(k) = bindings(k)._1 -> bindings(k)._2.typ
      result
    case TailLoop(name, args, resultType, _) => if (i == args.length)
        args.map { case (name, ir) => name -> ir.typ } :+
          name -> TTuple(TTuple(args.map(_._2.typ): _*), resultType)
      else empty
    case StreamMap(a, name, _) =>
      if (i == 1) Array(name -> tcoerce[TStream](a.typ).elementType) else empty
    case StreamZip(as, names, _, _, _) =>
      if (i == as.length) names.zip(as.map(a => tcoerce[TStream](a.typ).elementType)) else empty
    case StreamZipJoin(as, key, curKey, curVals, _) =>
      val eltType = tcoerce[TStruct](tcoerce[TStream](as.head.typ).elementType)
      if (i == as.length)
        Array(curKey -> eltType.typeAfterSelectNames(key), curVals -> TArray(eltType))
      else
        empty
    case StreamZipJoinProducers(contexts, ctxName, makeProducer, key, curKey, curVals, _) =>
      if (i == 1) {
        val contextType = elementType(contexts.typ)
        Array(ctxName -> contextType)
      } else if (i == 2) {
        val eltType = tcoerce[TStruct](elementType(makeProducer.typ))
        Array(curKey -> eltType.typeAfterSelectNames(key), curVals -> TArray(eltType))
      } else
        empty
    case StreamLeftIntervalJoin(left, right, _, _, lEltName, rEltName, _) =>
      if (i == 2)
        Array(lEltName -> elementType(left.typ), rEltName -> TArray(elementType(right.typ)))
      else empty
    case StreamFor(a, name, _) =>
      if (i == 1) Array(name -> tcoerce[TStream](a.typ).elementType) else empty
    case StreamFlatMap(a, name, _) =>
      if (i == 1) Array(name -> tcoerce[TStream](a.typ).elementType) else empty
    case StreamFilter(a, name, _) =>
      if (i == 1) Array(name -> tcoerce[TStream](a.typ).elementType) else empty
    case StreamTakeWhile(a, name, _) =>
      if (i == 1) Array(name -> tcoerce[TStream](a.typ).elementType) else empty
    case StreamDropWhile(a, name, _) =>
      if (i == 1) Array(name -> tcoerce[TStream](a.typ).elementType) else empty
    case StreamFold(a, zero, accumName, valueName, _) => if (i == 2)
        Array(accumName -> zero.typ, valueName -> tcoerce[TStream](a.typ).elementType)
      else empty
    case StreamFold2(a, accum, valueName, _, _) =>
      if (i <= accum.length)
        empty
      else if (i < 2 * accum.length + 1)
        Array((valueName, tcoerce[TStream](a.typ).elementType)) ++ accum.map { case (name, value) =>
          (name, value.typ)
        }
      else
        accum.map { case (name, value) => (name, value.typ) }
    case StreamBufferedAggregate(stream, _, _, _, name, _, _) =>
      if (i > 0) Array(name -> tcoerce[TStream](stream.typ).elementType) else empty
    case RunAggScan(a, name, _, _, _, _) =>
      if (i == 2 || i == 3) Array(name -> tcoerce[TStream](a.typ).elementType) else empty
    case StreamScan(a, zero, accumName, valueName, _) => if (i == 2)
        Array(accumName -> zero.typ, valueName -> tcoerce[TStream](a.typ).elementType)
      else empty
    case StreamAggScan(a, name, _) =>
      if (i == 1) FastSeq(name -> a.typ.asInstanceOf[TStream].elementType) else empty
    case StreamJoinRightDistinct(ll, rr, _, _, l, r, _, _) => if (i == 2)
        Array(l -> tcoerce[TStream](ll.typ).elementType, r -> tcoerce[TStream](rr.typ).elementType)
      else empty
    case ArraySort(a, left, right, _) => if (i == 1) Array(
        left -> tcoerce[TStream](a.typ).elementType,
        right -> tcoerce[TStream](a.typ).elementType,
      )
      else empty
    case ArrayMaximalIndependentSet(a, Some((left, right, _))) =>
      if (i == 1) {
        val typ = tcoerce[TArray](a.typ).elementType.asInstanceOf[TBaseStruct].types.head
        val tupleType = TTuple(typ)
        Array(left -> tupleType, right -> tupleType)
      } else {
        empty
      }
    case AggArrayPerElement(_, _, indexName, _, _, _) =>
      if (i == 1) FastSeq(indexName -> TInt32) else empty
    case AggFold(zero, _, _, accumName, otherAccumName, _) =>
      if (i == 1) FastSeq(accumName -> zero.typ)
      else if (i == 2) FastSeq(accumName -> zero.typ, otherAccumName -> zero.typ)
      else empty
    case NDArrayMap(nd, name, _) =>
      if (i == 1) Array(name -> tcoerce[TNDArray](nd.typ).elementType) else empty
    case NDArrayMap2(l, r, lName, rName, _, _) => if (i == 2) Array(
        lName -> tcoerce[TNDArray](l.typ).elementType,
        rName -> tcoerce[TNDArray](r.typ).elementType,
      )
      else empty
    case CollectDistributedArray(contexts, globals, cname, gname, _, _, _, _) => if (i == 2)
        Array(cname -> tcoerce[TStream](contexts.typ).elementType, gname -> globals.typ)
      else empty
    case TableAggregate(child, _) => if (i == 1) child.typ.globalEnv.m else empty
    case MatrixAggregate(child, _) => if (i == 1) child.typ.globalEnv.m else empty
    case TableFilter(child, _) => if (i == 1) child.typ.rowEnv.m else empty
    case TableGen(contexts, globals, cname, gname, _, _, _) =>
      if (i == 2) Array(cname -> elementType(contexts.typ), gname -> globals.typ)
      else empty
    case TableMapGlobals(child, _) => if (i == 1) child.typ.globalEnv.m else empty
    case TableMapRows(child, _) => if (i == 1) child.typ.rowEnv.m else empty
    case TableAggregateByKey(child, _) => if (i == 1) child.typ.globalEnv.m else empty
    case TableKeyByAndAggregate(child, _, _, _, _) =>
      if (i == 1) child.typ.globalEnv.m else if (i == 2) child.typ.rowEnv.m else empty
    case TableMapPartitions(child, g, p, _, _, _) =>
      if (i == 1) Array(g -> child.typ.globalType, p -> TStream(child.typ.rowType)) else empty
    case MatrixMapRows(child, _) => if (i == 1) child.typ.rowEnv.bind("n_cols", TInt32).m else empty
    case MatrixFilterRows(child, _) => if (i == 1) child.typ.rowEnv.m else empty
    case MatrixMapCols(child, _, _) =>
      if (i == 1) child.typ.colEnv.bind("n_rows", TInt64).m else empty
    case MatrixFilterCols(child, _) => if (i == 1) child.typ.colEnv.m else empty
    case MatrixMapEntries(child, _) => if (i == 1) child.typ.entryEnv.m else empty
    case MatrixFilterEntries(child, _) => if (i == 1) child.typ.entryEnv.m else empty
    case MatrixMapGlobals(child, _) => if (i == 1) child.typ.globalEnv.m else empty
    case MatrixAggregateColsByKey(child, _, _) =>
      if (i == 1) child.typ.rowEnv.m else if (i == 2) child.typ.globalEnv.m else empty
    case MatrixAggregateRowsByKey(child, _, _) =>
      if (i == 1) child.typ.colEnv.m else if (i == 2) child.typ.globalEnv.m else empty
    case BlockMatrixMap(_, eltName, _, _) => if (i == 1) Array(eltName -> TFloat64) else empty
    case BlockMatrixMap2(_, _, lName, rName, _, _) =>
      if (i == 2) Array(lName -> TFloat64, rName -> TFloat64) else empty
    case _ => empty
  }
}

object AggBindings {

  def apply(x: BaseIR, i: Int, parent: BindingEnv[_]): Option[Iterable[(String, Type)]] = {
    def wrapped(bindings: Iterable[(String, Type)]): Option[Iterable[(String, Type)]] = {
      if (parent.agg.isEmpty)
        throw new RuntimeException(s"aggEnv was None for child $i of $x")
      Some(bindings)
    }

    def base: Option[Iterable[(String, Type)]] = parent.agg.map(_ => FastSeq())

    x match {
      case AggLet(name, value, _, false) =>
        if (i == 1) wrapped(FastSeq(name -> value.typ)) else None
      case AggFilter(_, _, false) => if (i == 0) None else base
      case AggGroupBy(_, _, false) => if (i == 0) None else base
      case AggExplode(a, name, _, false) =>
        if (i == 1) wrapped(FastSeq(name -> a.typ.asInstanceOf[TIterable].elementType)) else None
      case AggArrayPerElement(a, elementName, indexName, _, _, false) => if (i == 1)
          wrapped(FastSeq(
            elementName -> a.typ.asInstanceOf[TIterable].elementType,
            indexName -> TInt32,
          ))
        else if (i == 2) base
        else None
      case StreamAgg(a, name, _) =>
        if (i == 1) Some(FastSeq(name -> a.typ.asInstanceOf[TIterable].elementType)) else base
      case TableAggregate(child, _) => if (i == 1) Some(child.typ.rowEnv.m) else None
      case MatrixAggregate(child, _) => if (i == 1) Some(child.typ.entryEnv.m) else None
      case RelationalLet(_, _, _) => None
      case CollectDistributedArray(_, _, _, _, _, _, _, _) if (i == 2) => None
      case _: ApplyAggOp => None
      case AggFold(_, _, _, _, _, false) => None
      case _: IR => base

      case TableAggregateByKey(child, _) => if (i == 1) Some(child.typ.rowEnv.m) else None
      case TableKeyByAndAggregate(child, _, _, _, _) =>
        if (i == 1) Some(child.typ.rowEnv.m) else None
      case _: TableIR => None

      case MatrixMapRows(child, _) => if (i == 1) Some(child.typ.entryEnv.m) else None
      case MatrixMapCols(child, _, _) => if (i == 1) Some(child.typ.entryEnv.m) else None
      case MatrixAggregateColsByKey(child, _, _) =>
        if (i == 1) Some(child.typ.entryEnv.m) else if (i == 2) Some(child.typ.colEnv.m) else None
      case MatrixAggregateRowsByKey(child, _, _) =>
        if (i == 1) Some(child.typ.entryEnv.m) else if (i == 2) Some(child.typ.rowEnv.m) else None
      case _: MatrixIR => None

      case _: BlockMatrixIR => None

    }
  }
}

object ScanBindings {
  def apply(x: BaseIR, i: Int, parent: BindingEnv[_]): Option[Iterable[(String, Type)]] = {
    def wrapped(bindings: Iterable[(String, Type)]): Option[Iterable[(String, Type)]] = {
      if (parent.scan.isEmpty)
        throw new RuntimeException(s"scanEnv was None for child $i of $x")
      Some(bindings)
    }

    def base: Option[Iterable[(String, Type)]] = parent.scan.map(_ => FastSeq())

    x match {
      case AggLet(name, value, _, true) => if (i == 1) wrapped(FastSeq(name -> value.typ)) else None
      case AggFilter(_, _, true) => if (i == 0) None else base
      case AggGroupBy(_, _, true) => if (i == 0) None else base
      case AggExplode(a, name, _, true) =>
        if (i == 1) wrapped(FastSeq(name -> a.typ.asInstanceOf[TIterable].elementType)) else None
      case AggArrayPerElement(a, elementName, indexName, _, _, true) => if (i == 1) wrapped(FastSeq(
          elementName -> a.typ.asInstanceOf[TIterable].elementType,
          indexName -> TInt32,
        ))
        else if (i == 2) base
        else None
      case AggFold(_, _, _, _, _, true) => None
      case StreamAggScan(a, name, _) =>
        if (i == 1) Some(FastSeq(name -> a.typ.asInstanceOf[TIterable].elementType)) else base
      case TableAggregate(_, _) => None
      case MatrixAggregate(_, _) => None
      case RelationalLet(_, _, _) => None
      case CollectDistributedArray(_, _, _, _, _, _, _, _) if (i == 2) => None
      case _: ApplyScanOp => None
      case _: IR => base

      case TableMapRows(child, _) => if (i == 1) Some(child.typ.rowEnv.m) else None
      case _: TableIR => None

      case MatrixMapRows(child, _) => if (i == 1) Some(child.typ.rowEnv.m) else None
      case MatrixMapCols(child, _, _) => if (i == 1) Some(child.typ.colEnv.m) else None
      case _: MatrixIR => None

      case _: BlockMatrixIR => None
    }
  }
}

object RelationalBindings {
  private val empty: Array[(String, Type)] = Array()

  def apply(x: BaseIR, i: Int): Iterable[(String, Type)] = {
    x match {
      case RelationalLet(name, value, _) => if (i == 1) FastSeq(name -> value.typ) else empty
      case RelationalLetTable(name, value, _) => if (i == 1) FastSeq(name -> value.typ) else empty
      case RelationalLetMatrixTable(name, value, _) =>
        if (i == 1) FastSeq(name -> value.typ) else empty
      case RelationalLetBlockMatrix(name, value, _) =>
        if (i == 1) FastSeq(name -> value.typ) else empty
      case _ => empty
    }
  }
}

object ChildEnvWithoutBindings {
  def apply[T](ir: BaseIR, i: Int, env: BindingEnv[T]): BindingEnv[T] = {
    ir match {
      case ArrayMaximalIndependentSet(_, Some(_)) if (i == 1) => env.copy(eval = Env.empty)
      case StreamAgg(_, _, _) => if (i == 1) env.createAgg else env
      case StreamAggScan(_, _, _) => if (i == 1) env.createScan else env
      case ApplyAggOp(init, _, _) => if (i < init.length) env.copy(agg = None) else env.promoteAgg
      case ApplyScanOp(init, _, _) =>
        if (i < init.length) env.copy(scan = None) else env.promoteScan
      case AggFold(_, _, _, _, _, isScan) => (isScan, i) match {
          case (true, 0) => env.noScan
          case (false, 0) => env.noAgg
          case (true, 1) => env.promoteScan
          case (false, 1) => env.promoteAgg
          case (true, 2) => env.copy(eval = Env.empty, scan = None)
          case (false, 2) => env.copy(eval = Env.empty, agg = None)
        }
      case CollectDistributedArray(_, _, _, _, _, _, _, _) =>
        if (i == 2) BindingEnv(relational = env.relational) else env
      case MatrixAggregate(_, _) => if (i == 0) env.onlyRelational()
        else BindingEnv(Env.empty, agg = Some(Env.empty), relational = env.relational)
      case TableAggregate(_, _) => if (i == 0) env.onlyRelational()
        else BindingEnv(Env.empty, agg = Some(Env.empty), relational = env.relational)
      case RelationalLet(_, _, _) =>
        if (i == 0) env.onlyRelational() else env.copy(agg = None, scan = None)
      case LiftMeOut(_) => BindingEnv(
          Env.empty[T],
          env.agg.map(_ => Env.empty),
          env.scan.map(_ => Env.empty),
          relational = env.relational,
        )
      case _: IR =>
        if (UsesAggEnv(ir, i)) env.promoteAgg else if (UsesScanEnv(ir, i)) env.promoteScan else env
      case x => BindingEnv(
          agg = AggBindings(x, i, env).map(_ => Env.empty),
          scan = ScanBindings(x, i, env).map(_ => Env.empty),
          relational = env.relational,
        )
    }
  }
}

case class Bindings2[A, B](
  childEnvWithoutBindings: BindingEnv[A],
  newBindings: BindingEnv[B],
) {
  def unified(implicit ev: BindingEnv[B] =:= BindingEnv[A]): BindingEnv[A] =
    childEnvWithoutBindings.merge(newBindings)

  def mapNewBindings[C](f: (String, B) => C): Bindings2[A, C] = Bindings2(
    childEnvWithoutBindings,
    newBindings.mapValuesWithKey(f),
  )

  def promoteAgg(isScan: Boolean): Bindings2[A, B] = Bindings2(
    childEnvWithoutBindings.promoteAggOrScan(isScan),
    newBindings.promoteAggOrScan(isScan),
  )

  def bindEval(bindings: (String, B)*): Bindings2[A, B] =
    copy(newBindings = newBindings.bindEval(bindings: _*))

  def dropEval: Bindings2[A, B] = Bindings2(
    childEnvWithoutBindings.copy(eval = Env.empty),
    newBindings.copy(eval = Env.empty),
  )

  def bindAgg(isScan: Boolean, bindings: (String, B)*): Bindings2[A, B] =
    copy(newBindings = newBindings.bindAggOrScan(isScan, bindings: _*))

  def createAgg(isScan: Boolean): Bindings2[A, B] = Bindings2(
    childEnvWithoutBindings.createAggOrScan(isScan),
    newBindings.createAggOrScan(isScan),
  )

  def noAgg(isScan: Boolean): Bindings2[A, B] = Bindings2(
    childEnvWithoutBindings.noAggOrScan(isScan),
    newBindings.noAggOrScan(isScan),
  )

  def onlyRelational(keepAggCapabilities: Boolean = false): Bindings2[A, B] = Bindings2(
    childEnvWithoutBindings.onlyRelational(keepAggCapabilities),
    newBindings.onlyRelational(keepAggCapabilities),
  )

  def bindRelational(bindings: (String, B)*): Bindings2[A, B] =
    copy(newBindings = newBindings.bindRelational(bindings: _*))
}

object Bindings2 {
  def empty[A, B]: Bindings2[A, B] = Bindings2(BindingEnv.empty, BindingEnv.empty)

  def apply[A](ir: BaseIR, i: Int, baseEnv: BindingEnv[A]): Bindings2[A, Type] = {
    val env = ChildEnvWithoutBindings(ir, i, baseEnv)
    val result = Bindings2(
      env,
      BindingEnv(
        Env.fromSeq(Bindings(ir, i)),
        AggBindings(ir, i, env).map(b => Env.fromSeq(b)),
        ScanBindings(ir, i, env).map(b => Env.fromSeq(b)),
        Env.fromSeq(RelationalBindings(ir, i)),
      ),
    )
    val newResult = apply2(ir, i, baseEnv)
    if (newResult != null)
      assert(
        newResult == result,
        s"mismatch in ir ${ir.getClass.getName} at child $i\nbaseEnv = $baseEnv\nold = $result\nnew = $newResult",
      )
    result
  }

  def apply2[A](ir: BaseIR, i: Int, _baseEnv: BindingEnv[A]): Bindings2[A, Type] = {
    val baseEnv = Bindings2[A, Type](_baseEnv, _baseEnv.dropBindings)

    ir match {
      case Let(bindings, _) =>
        val result = Array.ofDim[(String, Type)](i)
        for (k <- 0 until i) result(k) = bindings(k)._1 -> bindings(k)._2.typ
        baseEnv.bindEval(result: _*)
      case TailLoop(name, args, resultType, _) =>
        if (i == args.length)
          baseEnv
            .bindEval(args.map { case (name, ir) => name -> ir.typ }: _*)
            .bindEval(name -> TTuple(TTuple(args.map(_._2.typ): _*), resultType))
        else baseEnv
      case StreamMap(a, name, _) =>
        if (i == 1)
          baseEnv.bindEval(name -> elementType(a.typ))
        else baseEnv
      case StreamZip(as, names, _, _, _) =>
        if (i == as.length)
          baseEnv.bindEval(names.zip(as.map(a => elementType(a.typ))): _*)
        else baseEnv
      case StreamZipJoin(as, key, curKey, curVals, _) =>
        if (i == as.length) {
          val eltType = tcoerce[TStruct](elementType(as.head.typ))
          baseEnv.bindEval(
            curKey -> eltType.typeAfterSelectNames(key),
            curVals -> TArray(eltType),
          )
        } else baseEnv
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
      case StreamLeftIntervalJoin(left, right, _, _, lEltName, rEltName, _) =>
        if (i == 2) baseEnv.bindEval(
          lEltName -> elementType(left.typ),
          rEltName -> TArray(elementType(right.typ)),
        )
        else baseEnv
      case StreamFor(a, name, _) =>
        if (i == 1)
          baseEnv.bindEval(name -> elementType(a.typ))
        else baseEnv
      case StreamFlatMap(a, name, _) =>
        if (i == 1)
          baseEnv.bindEval(name -> elementType(a.typ))
        else baseEnv
      case StreamFilter(a, name, _) =>
        if (i == 1)
          baseEnv.bindEval(name -> elementType(a.typ))
        else baseEnv
      case StreamTakeWhile(a, name, _) =>
        if (i == 1)
          baseEnv.bindEval(name -> elementType(a.typ))
        else baseEnv
      case StreamDropWhile(a, name, _) =>
        if (i == 1)
          baseEnv.bindEval(name -> elementType(a.typ))
        else baseEnv
      case StreamFold(a, zero, accumName, valueName, _) =>
        if (i == 2)
          baseEnv.bindEval(accumName -> zero.typ, valueName -> elementType(a.typ))
        else baseEnv
      case StreamFold2(a, accum, valueName, _, _) =>
        if (i <= accum.length)
          baseEnv
        else if (i < 2 * accum.length + 1)
          baseEnv
            .bindEval(valueName -> elementType(a.typ))
            .bindEval(accum.map { case (name, value) => (name, value.typ) }: _*)
        else
          baseEnv.bindEval(accum.map { case (name, value) => (name, value.typ) }: _*)
      case StreamBufferedAggregate(stream, _, _, _, name, _, _) =>
        if (i > 0)
          baseEnv.bindEval(name -> elementType(stream.typ))
        else baseEnv
      case RunAggScan(a, name, _, _, _, _) =>
        if (i == 2 || i == 3)
          baseEnv.bindEval(name -> elementType(a.typ))
        else baseEnv
      case StreamScan(a, zero, accumName, valueName, _) =>
        if (i == 2)
          baseEnv.bindEval(
            accumName -> zero.typ,
            valueName -> elementType(a.typ),
          )
        else baseEnv
      case StreamAggScan(a, name, _) =>
        if (i == 1) {
          val eltType = elementType(a.typ)
          baseEnv
            .bindEval(name -> eltType)
            .createAgg(true).bindAgg(isScan = true, name -> eltType)
        } else baseEnv
      case StreamJoinRightDistinct(ll, rr, _, _, l, r, _, _) =>
        if (i == 2)
          baseEnv.bindEval(
            l -> elementType(ll.typ),
            r -> elementType(rr.typ),
          )
        else baseEnv
      case ArraySort(a, left, right, _) =>
        if (i == 1)
          baseEnv.bindEval(
            left -> elementType(a.typ),
            right -> elementType(a.typ),
          )
        else baseEnv
      case ArrayMaximalIndependentSet(a, Some((left, right, _))) =>
        if (i == 1) {
          val typ = tcoerce[TBaseStruct](elementType(a.typ)).types.head
          val tupleType = TTuple(typ)
          baseEnv.dropEval.bindEval(left -> tupleType, right -> tupleType)
        } else baseEnv
      case AggArrayPerElement(a, elementName, indexName, _, _, isScan) =>
        if (i == 0) baseEnv.promoteAgg(isScan)
        else if (i == 1)
          baseEnv
            .bindEval(indexName -> TInt32)
            .bindAgg(
              isScan,
              elementName -> elementType(a.typ),
              indexName -> TInt32,
            )
        else baseEnv
      case AggFold(zero, _, _, accumName, otherAccumName, isScan) =>
        if (i == 0) baseEnv.noAgg(isScan)
        else if (i == 1) baseEnv.promoteAgg(isScan).bindEval(accumName -> zero.typ)
        else baseEnv.dropEval.noAgg(isScan)
          .bindEval(accumName -> zero.typ, otherAccumName -> zero.typ)
      case NDArrayMap(nd, name, _) =>
        if (i == 1) baseEnv.bindEval(name -> tcoerce[TNDArray](nd.typ).elementType)
        else baseEnv
      case NDArrayMap2(l, r, lName, rName, _, _) =>
        if (i == 2) baseEnv.bindEval(
          lName -> tcoerce[TNDArray](l.typ).elementType,
          rName -> tcoerce[TNDArray](r.typ).elementType,
        )
        else baseEnv
      case CollectDistributedArray(contexts, globals, cname, gname, _, _, _, _) =>
        if (i == 2) baseEnv.onlyRelational().bindEval(
          cname -> elementType(contexts.typ),
          gname -> globals.typ,
        )
        else baseEnv
      case TableAggregate(child, _) =>
        if (i == 1)
          baseEnv.onlyRelational()
            .bindEval(child.typ.globalBindings: _*)
            .createAgg(false).bindAgg(isScan = false, child.typ.rowBindings: _*)
        else baseEnv.onlyRelational()
      case MatrixAggregate(child, _) =>
        if (i == 1)
          baseEnv.onlyRelational()
            .bindEval(child.typ.globalBindings: _*)
            .createAgg(false).bindAgg(isScan = false, child.typ.entryBindings: _*)
        else baseEnv.onlyRelational()
      case TableFilter(child, _) =>
        if (i == 1)
          baseEnv.onlyRelational().bindEval(child.typ.rowBindings: _*)
        else baseEnv.onlyRelational()
      case TableGen(contexts, globals, cname, gname, _, _, _) =>
        if (i == 2)
          baseEnv.onlyRelational().bindEval(
            cname -> elementType(contexts.typ),
            gname -> globals.typ,
          )
        else baseEnv.onlyRelational()
      case TableMapGlobals(child, _) =>
        if (i == 1)
          baseEnv.onlyRelational().bindEval(child.typ.globalBindings: _*)
        else baseEnv.onlyRelational()
      case TableMapRows(child, _) =>
        if (i == 1)
          baseEnv.onlyRelational()
            .bindEval(child.typ.rowBindings: _*)
            .createAgg(true).bindAgg(isScan = true, child.typ.rowBindings: _*)
        else baseEnv.onlyRelational()
      case TableAggregateByKey(child, _) =>
        if (i == 1)
          baseEnv.onlyRelational()
            .bindEval(child.typ.globalBindings: _*)
            .createAgg(false).bindAgg(isScan = false, child.typ.rowBindings: _*)
        else baseEnv.onlyRelational()
      case TableKeyByAndAggregate(child, _, _, _, _) =>
        if (i == 1)
          baseEnv.onlyRelational()
            .bindEval(child.typ.globalBindings: _*)
            .createAgg(false).bindAgg(isScan = false, child.typ.rowBindings: _*)
        else if (i == 2)
          baseEnv.onlyRelational().bindEval(child.typ.rowBindings: _*)
        else baseEnv.onlyRelational()
      case TableMapPartitions(child, g, p, _, _, _) =>
        if (i == 1)
          baseEnv.onlyRelational().bindEval(
            g -> child.typ.globalType,
            p -> TStream(child.typ.rowType),
          )
        else baseEnv.onlyRelational()
      case MatrixMapRows(child, _) =>
        if (i == 1)
          baseEnv.onlyRelational()
            .createAgg(false).createAgg(true)
            .bindEval(child.typ.rowBindings: _*)
            .bindEval("n_cols" -> TInt32)
            .bindAgg(isScan = false, child.typ.entryBindings: _*)
            .bindAgg(isScan = true, child.typ.rowBindings: _*)
        else baseEnv.onlyRelational()
      case MatrixFilterRows(child, _) =>
        if (i == 1)
          baseEnv.onlyRelational().bindEval(child.typ.rowBindings: _*)
        else baseEnv.onlyRelational()
      case MatrixMapCols(child, _, _) =>
        if (i == 1)
          baseEnv.onlyRelational()
            .createAgg(false).createAgg(true)
            .bindEval(child.typ.colBindings: _*)
            .bindEval("n_rows" -> TInt64)
            .bindAgg(isScan = false, child.typ.entryBindings: _*)
            .bindAgg(isScan = true, child.typ.colBindings: _*)
        else baseEnv.onlyRelational()
      case MatrixFilterCols(child, _) =>
        if (i == 1)
          baseEnv.onlyRelational().bindEval(child.typ.colBindings: _*)
        else baseEnv.onlyRelational()
      case MatrixMapEntries(child, _) =>
        if (i == 1)
          baseEnv.onlyRelational().bindEval(child.typ.entryBindings: _*)
        else baseEnv.onlyRelational()
      case MatrixFilterEntries(child, _) =>
        if (i == 1)
          baseEnv.onlyRelational().bindEval(child.typ.entryBindings: _*)
        else baseEnv.onlyRelational()
      case MatrixMapGlobals(child, _) =>
        if (i == 1)
          baseEnv.onlyRelational().bindEval(child.typ.globalBindings: _*)
        else baseEnv.onlyRelational()
      case MatrixAggregateColsByKey(child, _, _) =>
        if (i == 1)
          baseEnv.onlyRelational()
            .bindEval(child.typ.rowBindings: _*)
            .createAgg(false).bindAgg(isScan = false, child.typ.entryBindings: _*)
        else if (i == 2)
          baseEnv.onlyRelational()
            .bindEval(child.typ.globalBindings: _*)
            .createAgg(false).bindAgg(isScan = false, child.typ.colBindings: _*)
        else baseEnv.onlyRelational()
      case MatrixAggregateRowsByKey(child, _, _) =>
        if (i == 1)
          baseEnv.onlyRelational()
            .bindEval(child.typ.colBindings: _*)
            .createAgg(false).bindAgg(isScan = false, child.typ.entryBindings: _*)
        else if (i == 2)
          baseEnv.onlyRelational()
            .bindEval(child.typ.globalBindings: _*)
            .createAgg(false).bindAgg(isScan = false, child.typ.rowBindings: _*)
        else baseEnv.onlyRelational()
      case BlockMatrixMap(_, eltName, _, _) =>
        if (i == 1)
          baseEnv.onlyRelational().bindEval(eltName -> TFloat64)
        else baseEnv.onlyRelational()
      case BlockMatrixMap2(_, _, lName, rName, _, _) =>
        if (i == 2)
          baseEnv.onlyRelational()
            .bindEval(lName -> TFloat64, rName -> TFloat64)
        else baseEnv.onlyRelational()
      case AggLet(name, value, _, isScan) =>
        if (i == 0) baseEnv.promoteAgg(isScan)
        else baseEnv.bindAgg(isScan, name -> value.typ)
      case AggFilter(_, _, isScan) =>
        if (i == 0) baseEnv.promoteAgg(isScan)
        else baseEnv
      case AggGroupBy(_, _, isScan) =>
        if (i == 0) baseEnv.promoteAgg(isScan)
        else baseEnv
      case AggExplode(a, name, _, isScan) =>
        if (i == 0) baseEnv.promoteAgg(isScan)
        else baseEnv.bindAgg(isScan, name -> elementType(a.typ))
      case StreamAgg(a, name, _) =>
        if (i == 1)
          baseEnv.createAgg(isScan = false)
            .bindAgg(isScan = false, name -> elementType(a.typ))
        else baseEnv
      case RelationalLet(name, value, _) =>
        if (i == 1)
          baseEnv.noAgg(false).noAgg(true).bindRelational(name -> value.typ)
        else
          baseEnv.onlyRelational()
      case RelationalLetTable(name, value, _) =>
        if (i == 1)
          baseEnv.noAgg(false).noAgg(true).bindRelational(name -> value.typ)
        else
          baseEnv.onlyRelational()
      case RelationalLetMatrixTable(name, value, _) =>
        if (i == 1)
          baseEnv.noAgg(false).noAgg(true).bindRelational(name -> value.typ)
        else
          baseEnv.onlyRelational()
      case RelationalLetBlockMatrix(name, value, _) =>
        if (i == 1)
          baseEnv.noAgg(false).noAgg(true).bindRelational(name -> value.typ)
        else
          baseEnv.onlyRelational()
      case ApplyAggOp(init, _, _) =>
        if (i < init.length) baseEnv.noAgg(isScan = false)
        else baseEnv.promoteAgg(isScan = false)
      case ApplyScanOp(init, _, _) =>
        if (i < init.length) baseEnv.noAgg(isScan = true)
        else baseEnv.promoteAgg(isScan = true)
      case _: LiftMeOut =>
        baseEnv.onlyRelational(keepAggCapabilities = true)
      case _: IR =>
        if (UsesAggEnv(ir, i)) baseEnv.promoteAgg(false)
        else if (UsesScanEnv(ir, i)) baseEnv.promoteAgg(true)
        else baseEnv
      case _: TableIR | _: MatrixIR | _: BlockMatrixIR =>
        baseEnv.onlyRelational()
    }
  }
}

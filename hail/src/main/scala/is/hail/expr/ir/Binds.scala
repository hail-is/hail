package is.hail.expr.ir

import is.hail.types.tcoerce
import is.hail.types.virtual._
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
        name -> TTuple(TTuple(args.map(_._2.typ): _*), resultType) else empty
    case StreamMap(a, name, _) => if (i == 1) Array(name -> tcoerce[TStream](a.typ).elementType) else empty
    case StreamZip(as, names, _, _, _) => if (i == as.length) names.zip(as.map(a => tcoerce[TStream](a.typ).elementType)) else empty
    case StreamZipJoin(as, key, curKey, curVals, _) =>
      val eltType = tcoerce[TStruct](tcoerce[TStream](as.head.typ).elementType)
      if (i == as.length)
        Array(curKey -> eltType.typeAfterSelectNames(key),
              curVals -> TArray(eltType))
      else
        empty
    case StreamZipJoinProducers(contexts, ctxName, makeProducer, key, curKey, curVals, _) =>
      if (i == 1) {
        val contextType = TIterable.elementType(contexts.typ)
        Array(ctxName -> contextType)
      } else if (i == 2) {
        val eltType = tcoerce[TStruct](tcoerce[TStream](makeProducer.typ).elementType)
        Array(curKey -> eltType.typeAfterSelectNames(key),
          curVals -> TArray(eltType))
      } else
        empty
    case StreamFor(a, name, _) => if (i == 1) Array(name -> tcoerce[TStream](a.typ).elementType) else empty
    case StreamFlatMap(a, name, _) => if (i == 1) Array(name -> tcoerce[TStream](a.typ).elementType) else empty
    case StreamFilter(a, name, _) => if (i == 1) Array(name -> tcoerce[TStream](a.typ).elementType) else empty
    case StreamTakeWhile(a, name, _) => if (i == 1) Array(name -> tcoerce[TStream](a.typ).elementType) else empty
    case StreamDropWhile(a, name, _) => if (i == 1) Array(name -> tcoerce[TStream](a.typ).elementType) else empty
    case StreamFold(a, zero, accumName, valueName, _) => if (i == 2) Array(accumName -> zero.typ, valueName -> tcoerce[TStream](a.typ).elementType) else empty
    case StreamFold2(a, accum, valueName, seq, result) =>
      if (i <= accum.length)
        empty
      else if (i < 2 * accum.length + 1)
        Array((valueName, tcoerce[TStream](a.typ).elementType)) ++ accum.map { case (name, value) => (name, value.typ) }
      else
        accum.map { case (name, value) => (name, value.typ) }
    case StreamBufferedAggregate(stream, _,  _, _, name, _, _) => if (i > 0) Array(name -> tcoerce[TStream](stream.typ).elementType) else empty
    case RunAggScan(a, name, _, _, _, _) => if (i == 2 || i == 3) Array(name -> tcoerce[TStream](a.typ).elementType) else empty
    case StreamScan(a, zero, accumName, valueName, _) => if (i == 2) Array(accumName -> zero.typ, valueName -> tcoerce[TStream](a.typ).elementType) else empty
    case StreamAggScan(a, name, _) => if (i == 1) FastSeq(name -> a.typ.asInstanceOf[TStream].elementType) else empty
    case StreamJoinRightDistinct(ll, rr, _, _, l, r, _, _) => if (i == 2) Array(l -> tcoerce[TStream](ll.typ).elementType, r -> tcoerce[TStream](rr.typ).elementType) else empty
    case ArraySort(a, left, right, _) => if (i == 1) Array(left -> tcoerce[TStream](a.typ).elementType, right -> tcoerce[TStream](a.typ).elementType) else empty
    case ArrayMaximalIndependentSet(a, Some((left, right, _))) =>
      if (i == 1) {
        val typ = tcoerce[TArray](a.typ).elementType.asInstanceOf[TBaseStruct].types.head
        val tupleType = TTuple(typ)
        Array(left -> tupleType, right -> tupleType)
      } else {
        empty
      }
    case AggArrayPerElement(a, _, indexName, _, _, _) => if (i == 1) FastSeq(indexName -> TInt32) else empty
    case AggFold(zero, seqOp, combOp, accumName, otherAccumName, _) => {
      if (i == 1) FastSeq(accumName -> zero.typ)
      else if (i == 2) FastSeq(accumName -> zero.typ, otherAccumName -> zero.typ)
      else empty
    }
    case NDArrayMap(nd, name, _) => if (i == 1) Array(name -> tcoerce[TNDArray](nd.typ).elementType) else empty
    case NDArrayMap2(l, r, lName, rName, _, _) => if (i == 2) Array(lName -> tcoerce[TNDArray](l.typ).elementType, rName -> tcoerce[TNDArray](r.typ).elementType) else empty
    case CollectDistributedArray(contexts, globals, cname, gname, _, _, _, _) => if (i == 2) Array(cname -> tcoerce[TStream](contexts.typ).elementType, gname -> globals.typ) else empty
    case TableAggregate(child, _) => if (i == 1) child.typ.globalEnv.m else empty
    case MatrixAggregate(child, _) => if (i == 1) child.typ.globalEnv.m else empty
    case TableFilter(child, _) => if (i == 1) child.typ.rowEnv.m else empty
    case TableGen(contexts, globals, cname, gname, _, _, _) =>
      if (i == 2) Array(cname -> TIterable.elementType(contexts.typ), gname -> globals.typ)
      else empty
    case TableMapGlobals(child, _) => if (i == 1) child.typ.globalEnv.m else empty
    case TableMapRows(child, _) => if (i == 1) child.typ.rowEnv.m else empty
    case TableAggregateByKey(child, _) => if (i == 1) child.typ.globalEnv.m else empty
    case TableKeyByAndAggregate(child, _, _, _, _) => if (i == 1) child.typ.globalEnv.m else if (i == 2) child.typ.rowEnv.m else empty
    case TableMapPartitions(child, g, p, _, _, _) => if (i == 1) Array(g -> child.typ.globalType, p -> TStream(child.typ.rowType)) else empty
    case MatrixMapRows(child, _) => if (i == 1) child.typ.rowEnv.bind("n_cols", TInt32).m else empty
    case MatrixFilterRows(child, _) => if (i == 1) child.typ.rowEnv.m else empty
    case MatrixMapCols(child, _, _) => if (i == 1) child.typ.colEnv.bind("n_rows", TInt64).m else empty
    case MatrixFilterCols(child, _) => if (i == 1) child.typ.colEnv.m else empty
    case MatrixMapEntries(child, _) => if (i == 1) child.typ.entryEnv.m else empty
    case MatrixFilterEntries(child, _) => if (i == 1) child.typ.entryEnv.m else empty
    case MatrixMapGlobals(child, _) => if (i == 1) child.typ.globalEnv.m else empty
    case MatrixAggregateColsByKey(child, _, _) => if (i == 1) child.typ.rowEnv.m else if (i == 2) child.typ.globalEnv.m else empty
    case MatrixAggregateRowsByKey(child, _, _) => if (i == 1) child.typ.colEnv.m else if (i == 2) child.typ.globalEnv.m else empty
    case BlockMatrixMap(_, eltName, _, _) => if (i == 1) Array(eltName -> TFloat64) else empty
    case BlockMatrixMap2(_, _, lName, rName, _, _) => if (i == 2) Array(lName -> TFloat64, rName -> TFloat64) else empty
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
      case AggLet(name, value, _, false) => if (i == 1) wrapped(FastSeq(name -> value.typ)) else None
      case AggFilter(_, _, false) => if (i == 0) None else base
      case AggGroupBy(_, _, false) => if (i == 0) None else base
      case AggExplode(a, name, _, false) => if (i == 1) wrapped(FastSeq(name -> a.typ.asInstanceOf[TIterable].elementType)) else None
      case AggArrayPerElement(a, elementName, indexName, _, _, false) => if (i == 1) wrapped(FastSeq(elementName -> a.typ.asInstanceOf[TIterable].elementType, indexName -> TInt32)) else if (i == 2) base else None
      case StreamAgg(a, name, _) => if (i == 1) Some(FastSeq(name -> a.typ.asInstanceOf[TIterable].elementType)) else base
      case TableAggregate(child, _) => if (i == 1) Some(child.typ.rowEnv.m) else None
      case MatrixAggregate(child, _) => if (i == 1) Some(child.typ.entryEnv.m) else None
      case RelationalLet(_, _, _) => None
      case CollectDistributedArray(_, _, _, _, _, _, _, _) if (i == 2) => None
      case _: ApplyAggOp => None
      case AggFold(_, _, _, _, _, false) => None
      case _: IR => base

      case TableAggregateByKey(child, _) => if (i == 1) Some(child.typ.rowEnv.m) else None
      case TableKeyByAndAggregate(child, _, _, _, _) => if (i == 1) Some(child.typ.rowEnv.m) else None
      case _: TableIR => None

      case MatrixMapRows(child, _) => if (i == 1) Some(child.typ.entryEnv.m) else None
      case MatrixMapCols(child, _, _) => if (i == 1) Some(child.typ.entryEnv.m) else None
      case MatrixAggregateColsByKey(child, _, _) => if (i == 1) Some(child.typ.entryEnv.m) else if (i == 2) Some(child.typ.colEnv.m) else None
      case MatrixAggregateRowsByKey(child, _, _) => if (i == 1) Some(child.typ.entryEnv.m) else if (i == 2) Some(child.typ.rowEnv.m) else None
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
      case AggExplode(a, name, _, true) => if (i == 1) wrapped(FastSeq(name -> a.typ.asInstanceOf[TIterable].elementType)) else None
      case AggArrayPerElement(a, elementName, indexName, _, _, true) => if (i == 1) wrapped(FastSeq(elementName -> a.typ.asInstanceOf[TIterable].elementType, indexName -> TInt32)) else if (i == 2) base else None
      case AggFold(_, _, _, _, _, true) =>  None
      case StreamAggScan(a, name, _) => if (i == 1) Some(FastSeq(name -> a.typ.asInstanceOf[TIterable].elementType)) else base
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
      case RelationalLetMatrixTable(name, value, _) => if (i == 1) FastSeq(name -> value.typ) else empty
      case RelationalLetBlockMatrix(name, value, _) => if (i == 1) FastSeq(name -> value.typ) else empty
      case _ => empty
    }
  }
}

object NewBindings {
  def apply(x: BaseIR, i: Int, parent: BindingEnv[_]): BindingEnv[Type] = {
    BindingEnv(Env.fromSeq(Bindings(x, i)),
      agg = AggBindings(x, i, parent).map(b => Env.fromSeq(b)),
      scan = ScanBindings(x, i, parent).map(b => Env.fromSeq(b)),
      relational = Env.fromSeq(RelationalBindings(x, i)))
  }
}

object ChildEnvWithoutBindings {
  def apply[T](ir: BaseIR, i: Int, env: BindingEnv[T]): BindingEnv[T] = {
    ir match {
      case ArrayMaximalIndependentSet(_, Some(_)) if (i == 1) => env.copy(eval = Env.empty)
      case StreamAgg(_, _, _) => if (i == 1) env.createAgg else env
      case StreamAggScan(_, _, _) => if (i == 1) env.createScan else env
      case ApplyAggOp(init, _, _) => if (i < init.length) env.copy(agg = None) else env.promoteAgg
      case ApplyScanOp(init, _, _) => if (i < init.length) env.copy(scan = None) else env.promoteScan
      case AggFold(zero, seqOp, combOp, elementName, accumName, isScan) => (isScan, i) match {
        case (true, 0) => env.noScan
        case (false, 0) => env.noAgg
        case (true, 1) => env.promoteScan
        case (false, 1) => env.promoteAgg
        case (true, 2) => env.copy(eval = Env.empty, scan = None)
        case (false, 2) => env.copy(eval = Env.empty, agg = None)
      }
      case CollectDistributedArray(_, _, _, _, _, _, _, _) => if (i == 2) BindingEnv(relational = env.relational) else env
      case MatrixAggregate(_, _) => if (i == 0) env.onlyRelational else BindingEnv(Env.empty, agg = Some(Env.empty), relational = env.relational)
      case TableAggregate(_, _) => if (i == 0) env.onlyRelational else BindingEnv(Env.empty, agg = Some(Env.empty), relational = env.relational)
      case RelationalLet(_, _, _) => if (i == 0) env.onlyRelational else env.copy(agg = None, scan = None)
      case LiftMeOut(_) => BindingEnv(Env.empty[T], env.agg.map(_ => Env.empty), env.scan.map(_ => Env.empty), relational = env.relational)
      case _: IR => if (UsesAggEnv(ir, i)) env.promoteAgg else if (UsesScanEnv(ir, i)) env.promoteScan else env
      case x => BindingEnv(
        agg = AggBindings(x, i, env).map(_ => Env.empty),
        scan = ScanBindings(x, i, env).map(_ => Env.empty),
        relational = env.relational)
    }
  }
}

object ChildBindings {
  def apply(ir: BaseIR, i: Int, baseEnv: BindingEnv[Type]): BindingEnv[Type] = {
    val env = ChildEnvWithoutBindings(ir, i, baseEnv)
    val newBindings = NewBindings(ir, i, env)
    env.merge(newBindings)
  }

  def transformed[T](ir: BaseIR, i: Int, baseEnv: BindingEnv[T], f: (String, Type) => T): BindingEnv[T] = {
    val env = ChildEnvWithoutBindings(ir, i, baseEnv)
    val newBindings = NewBindings(ir, i, env).mapValuesWithKey(f)
    env.merge(newBindings)
  }
}

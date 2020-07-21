package is.hail.expr.ir

import is.hail.types.virtual._
import is.hail.utils._

object Binds {
  def apply(x: IR, v: String, i: Int): Boolean = Bindings(x, i).exists(_._1 == v)
}

object Bindings {
  private val empty: Array[(String, Type)] = Array()

  def apply(x: BaseIR, i: Int): Iterable[(String, Type)] = x match {
    case Let(name, value, _) => if (i == 1) Array(name -> value.typ) else empty
    case TailLoop(name, args, body) => if (i == args.length)
      args.map { case (name, ir) => name -> ir.typ } :+
        name -> TTuple(TTuple(args.map(_._2.typ): _*), body.typ) else empty
    case StreamMap(a, name, _) => if (i == 1) Array(name -> coerce[TStream](a.typ).elementType) else empty
    case StreamZip(as, names, _, _) => if (i == as.length) names.zip(as.map(a => coerce[TStream](a.typ).elementType)) else empty
    case StreamZipJoin(as, key, curKey, curVals, _) =>
      val eltType = coerce[TStruct](coerce[TStream](as.head.typ).elementType)
      if (i == as.length)
        Array(curKey -> eltType.typeAfterSelectNames(key),
              curVals -> TArray(eltType))
      else
        empty
    case StreamFor(a, name, _) => if (i == 1) Array(name -> coerce[TStream](a.typ).elementType) else empty
    case StreamFlatMap(a, name, _) => if (i == 1) Array(name -> coerce[TStream](a.typ).elementType) else empty
    case StreamFilter(a, name, _) => if (i == 1) Array(name -> coerce[TStream](a.typ).elementType) else empty
    case StreamFold(a, zero, accumName, valueName, _) => if (i == 2) Array(accumName -> zero.typ, valueName -> coerce[TStream](a.typ).elementType) else empty
    case StreamFold2(a, accum, valueName, seq, result) =>
      if (i <= accum.length)
        empty
      else if (i < 2 * accum.length + 1)
        Array((valueName, coerce[TStream](a.typ).elementType)) ++ accum.map { case (name, value) => (name, value.typ) }
      else
        accum.map { case (name, value) => (name, value.typ) }
    case RunAggScan(a, name, _, _, _, _) => if (i == 2 || i == 3) Array(name -> coerce[TStream](a.typ).elementType) else empty
    case StreamScan(a, zero, accumName, valueName, _) => if (i == 2) Array(accumName -> zero.typ, valueName -> coerce[TStream](a.typ).elementType) else empty
    case StreamAggScan(a, name, _) => if (i == 1) FastIndexedSeq(name -> a.typ.asInstanceOf[TStream].elementType) else empty
    case StreamJoinRightDistinct(ll, rr, _, _, l, r, _, _) => if (i == 2) Array(l -> coerce[TStream](ll.typ).elementType, r -> coerce[TStream](rr.typ).elementType) else empty
    case ArraySort(a, left, right, _) => if (i == 1) Array(left -> coerce[TStream](a.typ).elementType, right -> coerce[TStream](a.typ).elementType) else empty
    case AggArrayPerElement(a, _, indexName, _, _, _) => if (i == 1) FastIndexedSeq(indexName -> TInt32) else empty
    case NDArrayMap(nd, name, _) => if (i == 1) Array(name -> coerce[TNDArray](nd.typ).elementType) else empty
    case NDArrayMap2(l, r, lName, rName, _) => if (i == 2) Array(lName -> coerce[TNDArray](l.typ).elementType, rName -> coerce[TNDArray](r.typ).elementType) else empty
    case CollectDistributedArray(contexts, globals, cname, gname, _) => if (i == 2) Array(cname -> coerce[TStream](contexts.typ).elementType, gname -> globals.typ) else empty
    case TableAggregate(child, _) => if (i == 1) child.typ.globalEnv.m else empty
    case MatrixAggregate(child, _) => if (i == 1) child.typ.globalEnv.m else empty
    case TableFilter(child, _) => if (i == 1) child.typ.rowEnv.m else empty
    case TableMapGlobals(child, _) => if (i == 1) child.typ.globalEnv.m else empty
    case TableMapRows(child, _) => if (i == 1) child.typ.rowEnv.m else empty
    case TableAggregateByKey(child, _) => if (i == 1) child.typ.globalEnv.m else empty
    case TableKeyByAndAggregate(child, _, _, _, _) => if (i == 1) child.typ.globalEnv.m else if (i == 2) child.typ.rowEnv.m else empty
    case TableMapPartitions(child, g, p, _) => if (i == 1) Array(g -> child.typ.globalType, p -> TStream(child.typ.rowType)) else empty
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
    case x@ShuffleWith(_, _, _, _, name, _, _) =>
      if (i == 0 || i == 1) Array(name -> x.shuffleType) else empty
    case _ => empty
  }
}


object AggBindings {

  def apply(x: IR, i: Int, parent: BindingEnv[_]): Option[Iterable[(String, Type)]] = {
    def wrapped(bindings: Iterable[(String, Type)]): Option[Iterable[(String, Type)]] = {
      if (parent.agg.isEmpty)
        throw new RuntimeException(s"aggEnv was None for child $i of $x")
      Some(bindings)
    }

    def base: Option[Iterable[(String, Type)]] = parent.agg.map(_ => FastIndexedSeq())

    x match {
      case AggLet(name, value, _, false) => if (i == 1) wrapped(FastIndexedSeq(name -> value.typ)) else None
      case AggFilter(_, _, false) => if (i == 0) None else base
      case AggGroupBy(_, _, false) => if (i == 0) None else base
      case AggExplode(a, name, _, false) => if (i == 1) wrapped(FastIndexedSeq(name -> a.typ.asInstanceOf[TIterable].elementType)) else None
      case AggArrayPerElement(a, elementName, _, _, _, false) => if (i == 1) wrapped(FastIndexedSeq(elementName -> a.typ.asInstanceOf[TIterable].elementType)) else if (i == 2) base else None
      case StreamAgg(a, name, _) => if (i == 1) Some(FastIndexedSeq(name -> a.typ.asInstanceOf[TIterable].elementType)) else base
      case TableAggregate(child, _) => if (i == 1) Some(child.typ.rowEnv.m) else None
      case MatrixAggregate(child, _) => if (i == 1) Some(child.typ.entryEnv.m) else None
      case RelationalLet(_, _, _) => None
      case CollectDistributedArray(_, _, _, _, _) if (i == 2) => None
      case _: ApplyAggOp => None
      case _ => base
    }
  }

  def apply(x: TableIR, i: Int): Option[Iterable[(String, Type)]] = {
    x match {
      case TableAggregateByKey(child, _) => if (i == 1) Some(child.typ.rowEnv.m) else None
      case TableKeyByAndAggregate(child, _, _, _, _) => if (i == 1) Some(child.typ.rowEnv.m) else None
      case _ => None
    }
  }

  def apply(x: MatrixIR, i: Int): Option[Iterable[(String, Type)]] = x match {
    case MatrixMapRows(child, _) => if (i == 1) Some(child.typ.entryEnv.m) else None
    case MatrixMapCols(child, _, _) => if (i == 1) Some(child.typ.entryEnv.m) else None
    case MatrixAggregateColsByKey(child, _, _) => if (i == 1) Some(child.typ.entryEnv.m) else if (i == 2) Some(child.typ.colEnv.m) else None
    case MatrixAggregateRowsByKey(child, _, _) => if (i == 1) Some(child.typ.entryEnv.m) else if (i == 2) Some(child.typ.rowEnv.m) else None
    case _ => None
  }

  def apply(x: BlockMatrixIR, i: Int): Option[Iterable[(String, Type)]] = x match {
    case _ => None
  }
}

object ScanBindings {
  def apply(x: IR, i: Int, parent: BindingEnv[_]): Option[Iterable[(String, Type)]] = {
    def wrapped(bindings: Iterable[(String, Type)]): Option[Iterable[(String, Type)]] = {
      if (parent.scan.isEmpty)
        throw new RuntimeException(s"scanEnv was None for child $i of $x")
      Some(bindings)
    }

    def base: Option[Iterable[(String, Type)]] = parent.scan.map(_ => FastIndexedSeq())

    x match {
      case AggLet(name, value, _, true) => if (i == 1) wrapped(FastIndexedSeq(name -> value.typ)) else None
      case AggFilter(_, _, true) => if (i == 0) None else base
      case AggGroupBy(_, _, true) => if (i == 0) None else base
      case AggExplode(a, name, _, true) => if (i == 1) wrapped(FastIndexedSeq(name -> a.typ.asInstanceOf[TIterable].elementType)) else None
      case AggArrayPerElement(a, elementName, _, _, _, true) => if (i == 1) wrapped(FastIndexedSeq(elementName -> a.typ.asInstanceOf[TIterable].elementType)) else if (i == 2) base else None
      case StreamAggScan(a, name, _) => if (i == 1) Some(FastIndexedSeq(name -> a.typ.asInstanceOf[TIterable].elementType)) else base
      case TableAggregate(_, _) => None
      case MatrixAggregate(_, _) => None
      case RelationalLet(_, _, _) => None
      case CollectDistributedArray(_, _, _, _, _) if (i == 2) => None
      case _: ApplyScanOp => None
      case _ => base
    }
  }

  def apply(x: TableIR, i: Int): Option[Iterable[(String, Type)]] = {
    x match {
      case TableMapRows(child, _) => if (i == 1) Some(child.typ.rowEnv.m) else None
      case _ => None
    }
  }

  def apply(x: MatrixIR, i: Int): Option[Iterable[(String, Type)]] = x match {
    case MatrixMapRows(child, _) => if (i == 1) Some(child.typ.rowEnv.m) else None
    case MatrixMapCols(child, _, _) => if (i == 1) Some(child.typ.colEnv.m) else None
    case _ => None
  }

  def apply(x: BlockMatrixIR, i: Int): Option[Iterable[(String, Type)]] = x match {
    case _ => None
  }
}

object RelationalBindings {
  private val empty: Array[(String, Type)] = Array()

  def apply(x: IR, i: Int): Iterable[(String, Type)] = {
    x match {
      case RelationalLet(name, value, _) => if (i == 1) FastIndexedSeq(name -> value.typ) else empty
      case _ => empty
    }
  }

  def apply(x: TableIR, i: Int): Iterable[(String, Type)] = {
    x match {
      case RelationalLetTable(name, value, _) => if (i == 1) FastIndexedSeq(name -> value.typ) else empty
      case _ => empty
    }
  }

  def apply(x: MatrixIR, i: Int): Iterable[(String, Type)] =
    x match {
      case RelationalLetMatrixTable(name, value, _) => if (i == 1) FastIndexedSeq(name -> value.typ) else empty
      case _ => empty
    }

  def apply(x: BlockMatrixIR, i: Int): Iterable[(String, Type)] = x match {
    case RelationalLetBlockMatrix(name, value, _) => if (i == 1) FastIndexedSeq(name -> value.typ) else empty
    case _ => empty
  }
}

object NewBindings {
  def apply(x: IR, i: Int, parent: BindingEnv[_]): BindingEnv[Type] = {
    BindingEnv(Env.fromSeq(Bindings(x, i)),
      agg = AggBindings(x, i, parent).map(b => Env.fromSeq(b)),
      scan = ScanBindings(x, i, parent).map(b => Env.fromSeq(b)),
      relational = Env.fromSeq(RelationalBindings(x, i)))
  }

  def apply(x: TableIR, i: Int): BindingEnv[Type] = {
    BindingEnv(Env.fromSeq(Bindings(x, i)),
      agg = AggBindings(x, i).map(b => Env.fromSeq(b)),
      scan = ScanBindings(x, i).map(b => Env.fromSeq(b)),
      relational = Env.fromSeq(RelationalBindings(x, i)))
  }

  def apply(x: MatrixIR, i: Int): BindingEnv[Type] = {
    BindingEnv(Env.fromSeq(Bindings(x, i)),
      agg = AggBindings(x, i).map(b => Env.fromSeq(b)),
      scan = ScanBindings(x, i).map(b => Env.fromSeq(b)),
      relational = Env.fromSeq(RelationalBindings(x, i)))
  }

  def apply(x: BlockMatrixIR, i: Int): BindingEnv[Type] = {
    BindingEnv(Env.fromSeq(Bindings(x, i)),
      agg = AggBindings(x, i).map(b => Env.fromSeq(b)),
      scan = ScanBindings(x, i).map(b => Env.fromSeq(b)),
      relational = Env.fromSeq(RelationalBindings(x, i)))
  }
}

object ChildEnvWithoutBindings {
  def apply[T](ir: BaseIR, i: Int, env: BindingEnv[T]): BindingEnv[T] = {
    ir match {
      case StreamAgg(_, _, _) => if (i == 1) BindingEnv(eval = env.eval, agg = Some(env.eval), scan = env.scan.map(_ => Env.empty), relational = env.relational) else env
      case StreamAggScan(_, _, _) => if (i == 1) BindingEnv(eval = env.eval, agg = env.agg.map(_ => Env.empty), scan = Some(env.eval), relational = env.relational) else env
      case ApplyAggOp(init, _, _) => if (i < init.length) env.copy(agg = None) else env.promoteAgg
      case ApplyScanOp(init, _, _) => if (i < init.length) env.copy(scan = None) else env.promoteScan
      case CollectDistributedArray(_, _, _, _, _) => if (i == 2) BindingEnv(relational = env.relational) else env
      case MatrixAggregate(_, _) => if (i == 0) BindingEnv.empty else BindingEnv(Env.empty, agg = Some(Env.empty), relational = env.relational)
      case TableAggregate(_, _) => if (i == 0) BindingEnv.empty else BindingEnv(Env.empty, agg = Some(Env.empty), relational = env.relational)
      case RelationalLet(_, _, _) => if (i == 0) BindingEnv(relational = env.relational) else env.copy(agg = None, scan = None)
      case LiftMeOut(_) => BindingEnv(Env.empty[T], env.agg.map(_ => Env.empty), env.scan.map(_ => Env.empty), relational = env.relational)
      case tir: TableIR => BindingEnv(
        agg = AggBindings(tir, i).map(_ => Env.empty),
        scan = ScanBindings(tir, i).map(_ => Env.empty),
        relational = env.relational)
      case mir: MatrixIR =>
        BindingEnv(
          agg = AggBindings(mir, i).map(_ => Env.empty),
          scan = ScanBindings(mir, i).map(_ => Env.empty),
          relational = env.relational)
      case bmir: BlockMatrixIR =>
        BindingEnv(
          agg = AggBindings(bmir, i).map(_ => Env.empty),
          scan = ScanBindings(bmir, i).map(_ => Env.empty),
          relational = env.relational)
      case _: IR => if (UsesAggEnv(ir, i)) env.promoteAgg else if (UsesScanEnv(ir, i)) env.promoteScan else env
    }
  }
}

object ChildBindings {
  def apply(ir: IR, i: Int, baseEnv: BindingEnv[Type]): BindingEnv[Type] = {
    assert(ir.children(i).isInstanceOf[IR])
    val env = ChildEnvWithoutBindings(ir, i, baseEnv)
    val newBindings = NewBindings(ir, i, env)
    env.merge(newBindings)
  }

  def apply(ir: TableIR, i: Int): BindingEnv[Type] = NewBindings(ir, i)

  def apply(ir: MatrixIR, i: Int): BindingEnv[Type] = NewBindings(ir, i)

  def apply(ir: BlockMatrixIR, i: Int): BindingEnv[Type] = NewBindings(ir, i)
}

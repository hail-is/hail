package is.hail.expr.ir

import is.hail.expr.types.virtual._
import is.hail.utils._

object Binds {
  def apply(x: IR, v: String, i: Int): Boolean = Bindings(x, i).exists(_._1 == v)
}


object Bindings {
  private val empty: Array[(String, Type)] = Array()

  def apply(x: BaseIR, i: Int): Iterable[(String, Type)] = x match {
    case Let(name, value, _) => if (i == 1) Array(name -> value.typ) else empty
    case ArrayMap(a, name, _) => if (i == 1) Array(name -> -coerce[TStreamable](a.typ).elementType) else empty
    case ArrayFor(a, name, _) => if (i == 1) Array(name -> -coerce[TStreamable](a.typ).elementType) else empty
    case ArrayFlatMap(a, name, _) => if (i == 1) Array(name -> -coerce[TStreamable](a.typ).elementType) else empty
    case ArrayFilter(a, name, _) => if (i == 1) Array(name -> -coerce[TStreamable](a.typ).elementType) else empty
    case ArrayFold(a, zero, accumName, valueName, _) => if (i == 2) Array(accumName -> zero.typ, valueName -> -coerce[TStreamable](a.typ).elementType) else empty
    case ArrayScan(a, zero, accumName, valueName, _) => if (i == 2) Array(accumName -> zero.typ, valueName -> -coerce[TStreamable](a.typ).elementType) else empty
    case ArrayAggScan(a, name, _) => if (i == 1) FastIndexedSeq(name -> a.typ.asInstanceOf[TStreamable].elementType) else empty
    case ArrayLeftJoinDistinct(ll, rr, l, r, _, _) => if (i == 2 || i == 3) Array(l -> -coerce[TStreamable](ll.typ).elementType, r -> -coerce[TStreamable](rr.typ).elementType) else empty
    case ArraySort(a, left, right, _) => if (i == 1) Array(left -> -coerce[TStreamable](a.typ).elementType, right -> -coerce[TStreamable](a.typ).elementType) else empty
    case AggArrayPerElement(a, _, indexName, _, _, _) => if (i == 1) FastIndexedSeq(indexName -> TInt32()) else empty
    case NDArrayMap(nd, name, _) => if (i == 1) Array(name -> -coerce[TNDArray](nd.typ).elementType) else empty
    case NDArrayMap2(l, r, lName, rName, _) => if (i == 2) Array(lName -> -coerce[TNDArray](l.typ).elementType, rName -> -coerce[TNDArray](r.typ).elementType) else empty
    case CollectDistributedArray(contexts, globals, cname, gname, _) => if (i == 2) Array(cname -> -coerce[TArray](contexts.typ).elementType, gname -> globals.typ) else empty
    case Uniroot(argname, _, _, _) => if (i == 0) Array(argname -> TFloat64()) else empty
    case TableAggregate(child, _) => if (i == 1) child.typ.globalEnv.m else empty
    case MatrixAggregate(child, _) => if (i == 1) child.typ.globalEnv.m else empty
    case TableFilter(child, _) => if (i == 1) child.typ.rowEnv.m else empty
    case TableMapGlobals(child, _) => if (i == 1) child.typ.globalEnv.m else empty
    case TableMapRows(child, _) => if (i == 1) child.typ.rowEnv.m else empty
    case TableAggregateByKey(child, _) => if (i == 1) child.typ.globalEnv.m else empty
    case TableKeyByAndAggregate(child, _, _, _, _) => if (i == 1) child.typ.globalEnv.m else if (i == 2) child.typ.rowEnv.m else empty
    case MatrixMapRows(child, _) => if (i == 1) child.typ.rowEnv.m else empty
    case MatrixFilterRows(child, _) => if (i == 1) child.typ.rowEnv.m else empty
    case MatrixMapCols(child, _, _) => if (i == 1) child.typ.colEnv.m else empty
    case MatrixFilterCols(child, _) => if (i == 1) child.typ.colEnv.m else empty
    case MatrixMapEntries(child, _) => if (i == 1) child.typ.entryEnv.m else empty
    case MatrixFilterEntries(child, _) => if (i == 1) child.typ.entryEnv.m else empty
    case MatrixMapGlobals(child, _) => if (i == 1) child.typ.globalEnv.m else empty
    case MatrixAggregateColsByKey(child, _, _) => if (i == 1) child.typ.rowEnv.m else if (i == 2) child.typ.globalEnv.m else empty
    case MatrixAggregateRowsByKey(child, _, _) => if (i == 1) child.typ.colEnv.m else if (i == 2) child.typ.globalEnv.m else empty
    case BlockMatrixMap(_, _) => if (i == 1) Array("element" -> TFloat64()) else empty
    case BlockMatrixMap2(_, _, _) => if (i == 2) Array("l" -> TFloat64(), "r" -> TFloat64()) else empty
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
      case AggLet(name, value, _, false) => if (i == 1) wrapped(FastIndexedSeq(name -> value.typ)) else base
      case AggExplode(a, name, _, false) => if (i == 1) wrapped(FastIndexedSeq(name -> a.typ.asInstanceOf[TIterable].elementType)) else base
      case AggArrayPerElement(a, elementName, _, _, _, false) => if (i == 1) wrapped(FastIndexedSeq(elementName -> a.typ.asInstanceOf[TIterable].elementType)) else base
      case ArrayAgg(a, name, _) => if (i == 1) Some(FastIndexedSeq(name -> a.typ.asInstanceOf[TIterable].elementType)) else base
      case TableAggregate(child, _) => if (i == 1) wrapped(child.typ.rowEnv.m) else throw new UnsupportedOperationException
      case MatrixAggregate(child, _) => if (i == 1) Some(child.typ.entryEnv.m) else throw new UnsupportedOperationException
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
      case AggLet(name, value, _, true) => if (i == 1) wrapped(FastIndexedSeq(name -> value.typ)) else base
      case AggExplode(a, name, _, true) => if (i == 1) wrapped(FastIndexedSeq(name -> a.typ.asInstanceOf[TIterable].elementType)) else base
      case AggArrayPerElement(a, elementName, _, _, _, true) => if (i == 1) wrapped(FastIndexedSeq(elementName -> a.typ.asInstanceOf[TIterable].elementType)) else base
      case ArrayAggScan(a, name, _) => if (i == 1) Some(FastIndexedSeq(name -> a.typ.asInstanceOf[TIterable].elementType)) else base
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
    case MatrixMapRows(child, _) => if (i == 1) Some(child.typ.rowEnv.m) else throw new UnsupportedOperationException
    case MatrixMapCols(child, _, _) => if (i == 1) Some(child.typ.colEnv.m) else throw new UnsupportedOperationException
    case _ => None
  }

  def apply(x: BlockMatrixIR, i: Int): Option[Iterable[(String, Type)]] = x match {
    case _ => None
  }
}

object NewBindings {
  def apply(x: IR, i: Int, parent: BindingEnv[_]): BindingEnv[Type] = {
    BindingEnv(Env.fromSeq(Bindings(x, i)),
      agg = AggBindings(x, i, parent).map(b => Env.fromSeq(b)),
      scan = ScanBindings(x, i, parent).map(b => Env.fromSeq(b))
    )
  }

  def apply(x: TableIR, i: Int): BindingEnv[Type] = {
    BindingEnv(Env.fromSeq(Bindings(x, i)),
      agg = AggBindings(x, i).map(b => Env.fromSeq(b)),
      scan = ScanBindings(x, i).map(b => Env.fromSeq(b)))
  }

  def apply(x: MatrixIR, i: Int): BindingEnv[Type] = {
    BindingEnv(Env.fromSeq(Bindings(x, i)),
      agg = AggBindings(x, i).map(b => Env.fromSeq(b)),
      scan = ScanBindings(x, i).map(b => Env.fromSeq(b)))
  }

  def apply(x: BlockMatrixIR, i: Int): BindingEnv[Type] = {
    BindingEnv(Env.fromSeq(Bindings(x, i)),
      agg = AggBindings(x, i).map(b => Env.fromSeq(b)),
      scan = ScanBindings(x, i).map(b => Env.fromSeq(b)))
  }
}

object ChildEnvWithoutBindings {
  def apply[T](ir: IR, i: Int, env: BindingEnv[T]): BindingEnv[T] = {
    ir match {
      case ArrayAgg(_, _, _) => if (i == 1) BindingEnv(eval = env.eval, agg = Some(env.eval)) else env
      case ArrayAggScan(_, _, _) => if (i == 1) BindingEnv(eval = env.eval, scan = Some(env.eval)) else env
      case MatrixAggregate(_, _) => BindingEnv(Env.empty, agg = Some(Env.empty))
      case TableAggregate(_, _) => BindingEnv(Env.empty, agg = Some(Env.empty))
      case RelationalLet(_, _, _) => if (i == 0) BindingEnv.empty else env
      case _ => if (UsesAggEnv(ir, i)) env.promoteAgg else if (UsesScanEnv(ir, i)) env.promoteScan else env
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
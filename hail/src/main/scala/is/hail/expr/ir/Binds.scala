package is.hail.expr.ir

import is.hail.expr.types.virtual.{TArray, TContainer, TFloat64, Type}

object Binds {
  def apply(x: IR, v: String, i: Int): Boolean = Bindings(x, i).exists(_ == v)
}


object Bindings {
  private val empty: Array[(String, Type)] = Array()

  def apply(x: BaseIR, i: Int): Iterable[(String, Type)] = x match {
    case Let(name, value, _) => if (i == 1) Array(name -> value.typ) else empty
    case ArrayMap(a, name, _) => if (i == 1) Array(name -> -coerce[TArray](a.typ).elementType) else empty
    case ArrayFor(a, name, _) => if (i == 1) Array(name -> -coerce[TArray](a.typ).elementType) else empty
    case ArrayFlatMap(a, name, _) => if (i == 1) Array(name -> -coerce[TArray](a.typ).elementType) else empty
    case ArrayFilter(a, name, _) => if (i == 1) Array(name -> -coerce[TArray](a.typ).elementType) else empty
    case ArrayFold(a, zero, accumName, valueName, _) => if (i == 2) Array(accumName -> zero.typ, valueName -> -coerce[TArray](a.typ).elementType) else empty
    case ArrayScan(a, zero, accumName, valueName, _) => if (i == 2) Array(accumName -> zero.typ, valueName -> -coerce[TArray](a.typ).elementType) else empty
    case ArrayLeftJoinDistinct(ll, rr, l, r, _, _) => if (i == 2 || i == 3) Array(l -> -coerce[TArray](ll.typ).elementType, r -> -coerce[TArray](rr.typ).elementType) else empty
    case ArraySort(a, left, right, _) => if (i == 1) Array(left -> -coerce[TArray](a.typ).elementType, right -> -coerce[TArray](a.typ).elementType) else empty
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
    case _ => empty
  }
}

object AggBindings {
  private val empty: Array[(String, Type)] = Array()

  def apply(x: BaseIR, i: Int): Iterable[(String, Type)] = x match {
    case AggLet(name, value, _) => if (i == 1) Array(name -> value.typ) else empty
    case AggExplode(a, name, _) => if (i == 1) Array(name -> a.typ.asInstanceOf[TContainer].elementType) else empty
    case AggArrayPerElement(a, name, _) => if (i == 1) Array(name -> a.typ.asInstanceOf[TContainer].elementType) else empty
    case ArrayAgg(a, name, _) => if (i == 1) Array(name -> a.typ.asInstanceOf[TContainer].elementType) else empty
    case TableAggregate(child, _) => if (i == 1) child.typ.rowEnv.m else empty
    case MatrixAggregate(child, _) => if (i == 1) child.typ.entryEnv.m else empty
    case TableAggregateByKey(child, _) => if (i == 1) child.typ.rowEnv.m else empty
    case TableKeyByAndAggregate(child, _, _, _, _) => if (i == 1) child.typ.rowEnv.m else empty
    case MatrixMapRows(child, _) => if (i == 1) child.typ.entryEnv.m else empty
    case MatrixMapCols(child, _, _) => if (i == 1) child.typ.entryEnv.m else empty
    case MatrixAggregateColsByKey(child, _, _) => if (i == 1) child.typ.entryEnv.m else if (i == 2) child.typ.colEnv.m else empty
    case MatrixAggregateRowsByKey(child, _, _) => if (i == 1) child.typ.entryEnv.m else if (i == 2) child.typ.rowEnv.m else empty
    case _ => empty
  }
}

object ChildEnvWithoutBindings {
  private val empty = (Env.empty[Type], None)

  def apply(ir: BaseIR, i: Int, env: Env[Type], aggEnv: Option[Env[Type]]): (Env[Type], Option[Env[Type]]) = {
    ir match {
      case ArrayAgg(_, _, _) => if (i == 1) (env, Some(env)) else (env, aggEnv)
      case MatrixAggregate(_, _) => empty
      case TableAggregate(_, _) => empty
      case ArrayAgg(_, _, _) => if (i == 1) (env, Some(aggEnv.get.bindIterable(env.m))) else (env, aggEnv)
      case _ => (env, aggEnv)
    }
  }
}

object ChildEnvWithBindings {
  def apply(ir: BaseIR, i: Int, env: Env[Type], aggEnv: Option[Env[Type]]): (Env[Type], Option[Env[Type]]) = {
    val (baseEnv, baseAggEnv) = ChildEnvWithoutBindings(ir, i, env, aggEnv)
    assert(ir.children(i).isInstanceOf[IR])
    val b = Bindings(ir, i)
    val ab = AggBindings(ir, i)
    if (UsesAggEnv(ir, i)) {
      assert(b.isEmpty)
      if (baseAggEnv.isEmpty)
        throw new RuntimeException(s"$i: $ir")
      (baseAggEnv.get.bindIterable(ab), None)
    } else {
      (baseEnv.bindIterable(b), baseAggEnv match {
        case Some(ae) => Some(ae.bindIterable(ab))
        case None => if (ab.nonEmpty) Some(Env.empty.bindIterable(ab)) else None
      })

    }
  }
}

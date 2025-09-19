package is.hail.expr.ir

import is.hail.expr.ir.defs._
import is.hail.types.tcoerce
import is.hail.types.virtual._
import is.hail.types.virtual.TIterable.elementType
import is.hail.utils.FastSeq

import scala.collection.mutable

sealed abstract class AggEnv {
  def empty: AggEnv = this match {
    case AggEnv.Create(_) => AggEnv.Create(Seq.empty)
    case AggEnv.Bind(_) => AggEnv.NoOp
    case AggEnv.NoOp => AggEnv.NoOp
    case AggEnv.Drop => AggEnv.Drop
    case AggEnv.Promote => AggEnv.Promote
  }

  def isEmpty: Boolean = this match {
    case AggEnv.Create(bindings) => bindings.isEmpty
    case AggEnv.Bind(bindings) => bindings.isEmpty
    case _ => true
  }
}

object AggEnv {
  case object NoOp extends AggEnv
  case object Drop extends AggEnv
  case object Promote extends AggEnv
  final case class Create(bindings: Seq[Int]) extends AggEnv
  final case class Bind(bindings: Seq[Int]) extends AggEnv

  def bindOrNoOp(bindings: Seq[Int]): AggEnv =
    if (bindings.nonEmpty) Bind(bindings) else NoOp
}

object Binds {
  def apply(x: IR, v: Name, i: Int): Boolean = {
    val bindings = Bindings.get(x, i)
    bindings.all.zipWithIndex.exists { case ((name, _), i) =>
      name == v && bindings.eval.contains(i)
    }
  }
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
    Bindings(FastSeq.empty, FastSeq.empty, agg.empty, scan.empty, FastSeq.empty, dropEval)
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
    if (eval.isEmpty && agg.isEmpty && scan.isEmpty && relational.isEmpty)
      new Bindings(bindings, bindings.indices, agg, scan, relational, dropEval)
    else
      new Bindings(bindings, eval, agg, scan, relational, dropEval)

  val empty: Bindings[Nothing] =
    Bindings(FastSeq.empty, FastSeq.empty, AggEnv.NoOp, AggEnv.NoOp, FastSeq.empty, false)

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

  // Create a `Bindings` which cannot see anything bound in the enclosing context.
  private def inFreshScope(
    bindings: IndexedSeq[(Name, Type)] = FastSeq.empty,
    eval: IndexedSeq[Int] = FastSeq.empty,
    agg: Option[IndexedSeq[Int]] = None,
    scan: Option[IndexedSeq[Int]] = None,
    relational: IndexedSeq[Int] = FastSeq.empty,
  ): Bindings[Type] = Bindings(
    bindings,
    eval,
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
        val bindingsTypes = bindings.view.take(i).map(b => b.name -> b.value.typ).toArray
        val eval = mutable.ArrayBuilder.make[Int]
        val agg = mutable.ArrayBuilder.make[Int]
        val scan = mutable.ArrayBuilder.make[Int]
        for (k <- 0 until i) bindings(k) match {
          case Binding(_, _, Scope.EVAL) =>
            eval += k
          case Binding(_, _, Scope.AGG) =>
            agg += k
          case Binding(_, _, Scope.SCAN) =>
            scan += k
        }
        if (i < bindings.length) bindings(i).scope match {
          case Scope.EVAL =>
            Bindings(
              bindingsTypes,
              eval.result(),
              AggEnv.bindOrNoOp(agg.result()),
              AggEnv.bindOrNoOp(scan.result()),
            )
          case Scope.AGG =>
            Bindings(bindingsTypes, agg.result(), AggEnv.Promote, AggEnv.bindOrNoOp(scan.result()))
          case Scope.SCAN =>
            Bindings(bindingsTypes, scan.result(), AggEnv.bindOrNoOp(agg.result()), AggEnv.Promote)
        }
        else
          Bindings(
            bindingsTypes,
            eval.result(),
            AggEnv.bindOrNoOp(agg.result()),
            AggEnv.bindOrNoOp(scan.result()),
          )
      case TailLoop(name, args, resultType, _) if i == args.length =>
        Bindings(
          args.map { case (name, ir) => name -> ir.typ } :+
            name -> TTuple(TTuple(args.map(_._2.typ): _*), resultType)
        )
      case StreamMap(a, name, _) if i == 1 =>
        Bindings(FastSeq(name -> elementType(a.typ)))
      case StreamZip(as, names, _, _, _) if i == as.length =>
        Bindings(names.zip(as.map(a => elementType(a.typ))))
      case StreamZipJoin(as, key, curKey, curVals, _) if i == as.length =>
        val eltType = tcoerce[TStruct](elementType(as.head.typ))
        Bindings(FastSeq(
          curKey -> eltType.typeAfterSelectNames(key),
          curVals -> TArray(eltType),
        ))
      case StreamZipJoinProducers(contexts, ctxName, makeProducer, key, curKey, curVals, _) =>
        if (i == 1) {
          val contextType = elementType(contexts.typ)
          Bindings(FastSeq(ctxName -> contextType))
        } else if (i == 2) {
          val eltType = tcoerce[TStruct](elementType(makeProducer.typ))
          Bindings(FastSeq(
            curKey -> eltType.typeAfterSelectNames(key),
            curVals -> TArray(eltType),
          ))
        } else Bindings.empty
      case StreamLeftIntervalJoin(left, right, _, _, lEltName, rEltName, _) if i == 2 =>
        Bindings(FastSeq(
          lEltName -> elementType(left.typ),
          rEltName -> TArray(elementType(right.typ)),
        ))
      case StreamFor(a, name, _) if i == 1 =>
        Bindings(FastSeq(name -> elementType(a.typ)))
      case StreamFlatMap(a, name, _) if i == 1 =>
        Bindings(FastSeq(name -> elementType(a.typ)))
      case StreamFilter(a, name, _) if i == 1 =>
        Bindings(FastSeq(name -> elementType(a.typ)))
      case StreamTakeWhile(a, name, _) if i == 1 =>
        Bindings(FastSeq(name -> elementType(a.typ)))
      case StreamDropWhile(a, name, _) if i == 1 =>
        Bindings(FastSeq(name -> elementType(a.typ)))
      case StreamFold(a, zero, accumName, valueName, _) if i == 2 =>
        Bindings(FastSeq(accumName -> zero.typ, valueName -> elementType(a.typ)))
      case StreamFold2(a, accum, valueName, _, _) =>
        if (i < accum.length + 1)
          Bindings.empty
        else if (i < 2 * accum.length + 1)
          Bindings(
            (valueName -> elementType(a.typ)) +:
              accum.map { case (name, value) => (name, value.typ) }
          )
        else
          Bindings(accum.map { case (name, value) => (name, value.typ) })
      case StreamBufferedAggregate(stream, _, _, _, name, _, _) if i > 0 =>
        Bindings(FastSeq(name -> elementType(stream.typ)))
      case RunAggScan(a, name, _, _, _, _) if i == 2 || i == 3 =>
        Bindings(FastSeq(name -> elementType(a.typ)))
      case StreamAgg(a, name, _) if i == 1 =>
        Bindings(
          FastSeq(name -> elementType(a.typ)),
          agg = AggEnv.Create(FastSeq(0)),
        )
      case StreamScan(a, zero, accumName, valueName, _) if i == 2 =>
        Bindings(FastSeq(accumName -> zero.typ, valueName -> elementType(a.typ)))
      case StreamAggScan(a, name, _) if i == 1 =>
        val eltType = elementType(a.typ)
        Bindings(
          FastSeq(name -> eltType),
          eval = FastSeq(0),
          scan = AggEnv.Create(FastSeq(0)),
        )
      case StreamJoinRightDistinct(ll, rr, _, _, l, r, _, _) if i == 2 =>
        Bindings(FastSeq(
          l -> elementType(ll.typ),
          r -> elementType(rr.typ),
        ))
      case ArraySort(a, left, right, _) if i == 1 =>
        Bindings(FastSeq(
          left -> elementType(a.typ),
          right -> elementType(a.typ),
        ))
      case ArrayMaximalIndependentSet(a, Some((left, right, _))) if i == 1 =>
        val typ = tcoerce[TBaseStruct](elementType(a.typ)).types.head
        val tupleType = TTuple(typ)
        Bindings(FastSeq(left -> tupleType, right -> tupleType), dropEval = true)
      case AggArrayPerElement(a, elementName, indexName, _, _, isScan) =>
        if (i == 0)
          Bindings(
            agg = if (isScan) AggEnv.NoOp else AggEnv.Promote,
            scan = if (!isScan) AggEnv.NoOp else AggEnv.Promote,
          )
        else if (i == 1) {
          Bindings(
            FastSeq(
              elementName -> elementType(a.typ),
              indexName -> TInt32,
            ),
            eval = FastSeq(1),
            agg = if (isScan) AggEnv.NoOp else AggEnv.Bind(FastSeq(0, 1)),
            scan = if (!isScan) AggEnv.NoOp else AggEnv.Bind(FastSeq(0, 1)),
          )
        } else Bindings.empty
      case AggFold(zero, _, _, accumName, otherAccumName, isScan) =>
        if (i == 0)
          Bindings(
            agg = if (isScan) AggEnv.NoOp else AggEnv.Drop,
            scan = if (!isScan) AggEnv.NoOp else AggEnv.Drop,
          )
        else if (i == 1)
          Bindings(
            FastSeq(accumName -> zero.typ),
            agg = if (isScan) AggEnv.NoOp else AggEnv.Promote,
            scan = if (!isScan) AggEnv.NoOp else AggEnv.Promote,
          )
        else
          Bindings(
            FastSeq(accumName -> zero.typ, otherAccumName -> zero.typ),
            agg = if (isScan) AggEnv.NoOp else AggEnv.Drop,
            scan = if (!isScan) AggEnv.NoOp else AggEnv.Drop,
            dropEval = true,
          )
      case NDArrayMap(nd, name, _) if i == 1 =>
        Bindings(FastSeq(name -> tcoerce[TNDArray](nd.typ).elementType))
      case NDArrayMap2(l, r, lName, rName, _, _) if i == 2 =>
        Bindings(FastSeq(
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
        if (i == 1)
          Bindings.inFreshScope(
            child.typ.rowBindings,
            eval = TableType.globalBindings,
            agg = Some(TableType.rowBindings),
          )
        else Bindings(agg = AggEnv.Drop, scan = AggEnv.Drop, dropEval = true)
      case MatrixAggregate(child, _) =>
        if (i == 1)
          Bindings.inFreshScope(
            child.typ.entryBindings,
            eval = MatrixType.globalBindings,
            agg = Some(MatrixType.entryBindings),
          )
        else Bindings(agg = AggEnv.Drop, scan = AggEnv.Drop, dropEval = true)
      case ApplyAggOp(init, _, _) =>
        if (i < init.length) Bindings(agg = AggEnv.Drop)
        else Bindings(agg = AggEnv.Promote)
      case ApplyScanOp(init, _, _) =>
        if (i < init.length) Bindings(scan = AggEnv.Drop)
        else Bindings(scan = AggEnv.Promote)
      case AggFilter(_, _, isScan) if i == 0 =>
        Bindings(
          agg = if (isScan) AggEnv.NoOp else AggEnv.Promote,
          scan = if (!isScan) AggEnv.NoOp else AggEnv.Promote,
        )
      case AggGroupBy(_, _, isScan) if i == 0 =>
        Bindings(
          agg = if (isScan) AggEnv.NoOp else AggEnv.Promote,
          scan = if (!isScan) AggEnv.NoOp else AggEnv.Promote,
        )
      case AggExplode(a, name, _, isScan) =>
        if (i == 0)
          Bindings(
            agg = if (isScan) AggEnv.NoOp else AggEnv.Promote,
            scan = if (!isScan) AggEnv.NoOp else AggEnv.Promote,
          )
        else
          Bindings(
            FastSeq(name -> elementType(a.typ)),
            agg = if (isScan) AggEnv.NoOp else AggEnv.Bind(FastSeq(0)),
            scan = if (!isScan) AggEnv.NoOp else AggEnv.Bind(FastSeq(0)),
          )
      case RelationalLet(name, value, _) =>
        if (i == 1)
          Bindings(
            FastSeq(name -> value.typ),
            agg = AggEnv.Drop,
            scan = AggEnv.Drop,
            relational = FastSeq(0),
          )
        else
          Bindings(
            agg = AggEnv.Drop,
            scan = AggEnv.Drop,
            dropEval = true,
          )
      case _: LiftMeOut =>
        Bindings(
          agg = AggEnv.Drop,
          scan = AggEnv.Drop,
          dropEval = true,
        )
      case _ =>
        if (UsesAggEnv(ir, i))
          Bindings(agg = AggEnv.Promote)
        else if (UsesScanEnv(ir, i))
          Bindings(scan = AggEnv.Promote)
        else Bindings.empty
    }
}

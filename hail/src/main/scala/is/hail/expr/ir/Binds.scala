package is.hail.expr.ir

import is.hail.types.tcoerce
import is.hail.types.virtual._
import is.hail.types.virtual.TIterable.elementType
import is.hail.utils.FastSeq

import scala.collection.mutable

object Binds {
  def apply(x: IR, v: String, i: Int): Boolean =
    Bindings.get(x, i).eval.exists(_._1 == v)
}

final case class Bindings[+T](
  eval: IndexedSeq[(String, T)] = FastSeq.empty,
  agg: AggEnv[T] = AggEnv.NoOp,
  scan: AggEnv[T] = AggEnv.NoOp,
  relational: IndexedSeq[(String, T)] = FastSeq.empty,
  dropEval: Boolean = false,
) {
  def map[U](f: (String, T) => U): Bindings[U] = Bindings(
    eval.map { case (n, v) => n -> f(n, v) },
    agg.map(f),
    scan.map(f),
    relational.map { case (n, v) => n -> f(n, v) },
    dropEval,
  )

  def allEmpty: Boolean =
    eval.isEmpty && agg.isEmpty && scan.isEmpty && relational.isEmpty

  def dropBindings[U]: Bindings[U] =
    Bindings(FastSeq.empty, agg.empty, scan.empty, FastSeq.empty, dropEval)
}

object Bindings {
  val empty: Bindings[Nothing] =
    Bindings(FastSeq.empty, AggEnv.NoOp, AggEnv.NoOp, FastSeq.empty, false)

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
    eval: IndexedSeq[(String, Type)] = FastSeq.empty,
    agg: Option[IndexedSeq[(String, Type)]] = None,
    scan: Option[IndexedSeq[(String, Type)]] = None,
    relational: IndexedSeq[(String, Type)] = FastSeq.empty,
  ): Bindings[Type] = Bindings(
    eval,
    agg.map(AggEnv.Create(_)).getOrElse(AggEnv.Drop),
    scan.map(AggEnv.Create(_)).getOrElse(AggEnv.Drop),
    relational,
    dropEval = true,
  )

  private def childEnvMatrix(ir: MatrixIR, i: Int): Bindings[Type] = {
    ir match {
      case MatrixMapRows(child, _) if i == 1 =>
        Bindings.inFreshScope(
          eval = child.typ.rowBindings :+ "n_cols" -> TInt32,
          agg = Some(child.typ.entryBindings),
          scan = Some(child.typ.rowBindings),
        )
      case MatrixFilterRows(child, _) if i == 1 =>
        Bindings.inFreshScope(child.typ.rowBindings)
      case MatrixMapCols(child, _, _) if i == 1 =>
        Bindings.inFreshScope(
          eval = child.typ.colBindings :+ "n_rows" -> TInt64,
          agg = Some(child.typ.entryBindings),
          scan = Some(child.typ.colBindings),
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
        if (i == 1) {
          Bindings.inFreshScope(
            eval = child.typ.rowBindings,
            agg = Some(child.typ.entryBindings),
          )
        } else if (i == 2) {
          Bindings.inFreshScope(
            eval = child.typ.globalBindings,
            agg = Some(child.typ.colBindings),
          )
        } else Bindings.inFreshScope()
      case MatrixAggregateRowsByKey(child, _, _) =>
        if (i == 1)
          Bindings.inFreshScope(
            eval = child.typ.colBindings,
            agg = Some(child.typ.entryBindings),
          )
        else if (i == 2)
          Bindings.inFreshScope(
            eval = child.typ.globalBindings,
            agg = Some(child.typ.rowBindings),
          )
        else Bindings.inFreshScope()
      case RelationalLetMatrixTable(name, value, _) if i == 1 =>
        Bindings.inFreshScope(relational = FastSeq(name -> value.typ))
      case _ =>
        Bindings.inFreshScope()
    }
  }

  private def childEnvTable(ir: TableIR, i: Int): Bindings[Type] = {
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
        Bindings.inFreshScope(
          eval = child.typ.rowBindings,
          scan = Some(child.typ.rowBindings),
        )
      case TableAggregateByKey(child, _) if i == 1 =>
        Bindings.inFreshScope(
          eval = child.typ.globalBindings,
          agg = Some(child.typ.rowBindings),
        )
      case TableKeyByAndAggregate(child, _, _, _, _) =>
        if (i == 1)
          Bindings.inFreshScope(
            eval = child.typ.globalBindings,
            agg = Some(child.typ.rowBindings),
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
        Bindings.inFreshScope(relational = FastSeq(name -> value.typ))
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
      case RelationalLetBlockMatrix(name, value, _) if i == 1 =>
        Bindings.inFreshScope(relational = FastSeq(name -> value.typ))
      case _ =>
        Bindings.inFreshScope()
    }
  }

  private def childEnvValue(ir: IR, i: Int): Bindings[Type] =
    ir match {
      case Block(bindings, _) =>
        val eval = mutable.ArrayBuilder.make[(String, Type)]
        val agg = mutable.ArrayBuilder.make[(String, Type)]
        val scan = mutable.ArrayBuilder.make[(String, Type)]
        for (k <- 0 until i) bindings(k) match {
          case Binding(name, value, Scope.EVAL) =>
            eval += name -> value.typ
          case Binding(name, value, Scope.AGG) =>
            agg += name -> value.typ
          case Binding(name, value, Scope.SCAN) =>
            scan += name -> value.typ
        }
        if (i < bindings.length) bindings(i).scope match {
          case Scope.EVAL =>
            Bindings(
              eval.result(),
              AggEnv.bindOrNoOp(agg.result()),
              AggEnv.bindOrNoOp(scan.result()),
            )
          case Scope.AGG => Bindings(agg.result(), AggEnv.Promote, AggEnv.bindOrNoOp(scan.result()))
          case Scope.SCAN =>
            Bindings(scan.result(), AggEnv.bindOrNoOp(agg.result()), AggEnv.Promote)
        }
        else
          Bindings(eval.result(), AggEnv.bindOrNoOp(agg.result()), AggEnv.bindOrNoOp(scan.result()))
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
        if (i <= accum.length)
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
      case StreamScan(a, zero, accumName, valueName, _) if i == 2 =>
        Bindings(FastSeq(
          accumName -> zero.typ,
          valueName -> elementType(a.typ),
        ))
      case StreamAggScan(a, name, _) if i == 1 =>
        val eltType = elementType(a.typ)
        Bindings(
          eval = FastSeq(name -> eltType),
          scan = AggEnv.Create(FastSeq(name -> eltType)),
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
            eval = FastSeq(indexName -> TInt32),
            agg = if (isScan) AggEnv.NoOp
            else AggEnv.Bind(FastSeq(
              elementName -> elementType(a.typ),
              indexName -> TInt32,
            )),
            scan = if (!isScan) AggEnv.NoOp
            else AggEnv.Bind(FastSeq(
              elementName -> elementType(a.typ),
              indexName -> TInt32,
            )),
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
            eval = FastSeq(accumName -> zero.typ),
            agg = if (isScan) AggEnv.NoOp else AggEnv.Promote,
            scan = if (!isScan) AggEnv.NoOp else AggEnv.Promote,
          )
        else
          Bindings(
            eval = FastSeq(accumName -> zero.typ, otherAccumName -> zero.typ),
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
        Bindings(
          eval = FastSeq(
            cname -> elementType(contexts.typ),
            gname -> globals.typ,
          ),
          agg = AggEnv.Drop,
          scan = AggEnv.Drop,
          dropEval = true,
        )
      case TableAggregate(child, _) =>
        if (i == 1)
          Bindings(
            eval = child.typ.globalBindings,
            agg = AggEnv.Create(child.typ.rowBindings),
            scan = AggEnv.Drop,
            dropEval = true,
          )
        else Bindings(agg = AggEnv.Drop, scan = AggEnv.Drop, dropEval = true)
      case MatrixAggregate(child, _) =>
        if (i == 1)
          Bindings(
            eval = child.typ.globalBindings,
            agg = AggEnv.Create(child.typ.entryBindings),
            scan = AggEnv.Drop,
            dropEval = true,
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
            agg = if (isScan) AggEnv.NoOp else AggEnv.Bind(FastSeq(name -> elementType(a.typ))),
            scan = if (!isScan) AggEnv.NoOp else AggEnv.Bind(FastSeq(name -> elementType(a.typ))),
          )
      case StreamAgg(a, name, _) if i == 1 =>
        Bindings(agg = AggEnv.Create(FastSeq(name -> elementType(a.typ))))
      case RelationalLet(name, value, _) =>
        if (i == 1)
          Bindings(
            agg = AggEnv.Drop,
            scan = AggEnv.Drop,
            relational = FastSeq(name -> value.typ),
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

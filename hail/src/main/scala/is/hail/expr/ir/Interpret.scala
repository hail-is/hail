package is.hail.expr.ir

import is.hail.annotations._
import is.hail.annotations.aggregators.RegionValueAggregator
import is.hail.asm4s.AsmFunction3
import is.hail.expr.TypedAggregator
import is.hail.expr.ir.lowering.LoweringPipeline
import is.hail.expr.types.physical.{PTuple, PType}
import is.hail.expr.types.virtual._
import is.hail.io.BufferSpec
import is.hail.linalg.BlockMatrix
import is.hail.methods._
import is.hail.rvd.RVDContext
import is.hail.utils._
import is.hail.{HailContext, stats}
import org.apache.spark.sql.Row

object Interpret {
  type Agg = (IndexedSeq[Row], TStruct)

  def apply(tir: TableIR, ctx: ExecuteContext): TableValue =
    apply(tir, ctx, optimize = true)

  def apply(tir: TableIR, ctx: ExecuteContext, optimize: Boolean): TableValue = {
    val lowered = LoweringPipeline.legacyRelationalLowerer(ctx, tir, optimize).asInstanceOf[TableIR]
    lowered.execute(ctx)
  }

  def apply(mir: MatrixIR, ctx: ExecuteContext, optimize: Boolean): TableValue = {
    val lowered = LoweringPipeline.legacyRelationalLowerer(ctx, mir, optimize).asInstanceOf[TableIR]
    lowered.execute(ctx)
  }

  def apply(bmir: BlockMatrixIR, ctx: ExecuteContext, optimize: Boolean): BlockMatrix = {
    val lowered = LoweringPipeline.legacyRelationalLowerer(ctx, bmir, optimize).asInstanceOf[BlockMatrixIR]
    lowered.execute(ctx)
  }

  def apply[T](ctx: ExecuteContext, ir: IR): T = apply(ctx, ir, Env.empty[(Any, Type)], FastIndexedSeq(), None).asInstanceOf[T]

  def apply[T](ctx: ExecuteContext,
    ir0: IR,
    env: Env[(Any, Type)],
    args: IndexedSeq[(Any, Type)],
    aggArgs: Option[Agg],
    optimize: Boolean = true
  ): T = {
    val (typeEnv, valueEnv) = env.m.foldLeft((Env.empty[Type], Env.empty[Any])) {
      case ((e1, e2), (k, (value, t))) => (e1.bind(k, t), e2.bind(k, value))
    }

    var ir = ir0.unwrap

    def optimizeIR(context: String) {
      ir = Optimize(ir, noisy = true, context = context, ctx)
      TypeCheck(ir, BindingEnv(typeEnv, agg = aggArgs.map { agg =>
        agg._2.fields.foldLeft(Env.empty[Type]) { case (env, f) =>
          env.bind(f.name, f.typ)
        }
      }))
    }

    if (optimize) optimizeIR("Interpret, first pass")
    ir = LowerMatrixIR(ir)
    if (optimize) optimizeIR("Interpret, after lowering MatrixIR")
    ir = InterpretNonCompilable(ctx, ir).asInstanceOf[IR]

    val result = apply(ctx, ir, valueEnv, args, aggArgs, None, Memo.empty).asInstanceOf[T]

    result
  }

  def alreadyLowered(ctx: ExecuteContext, ir: IR): Any = apply(ctx, ir, Env.empty, FastIndexedSeq(), None, None, Memo.empty)

  private def apply(ctx: ExecuteContext,
    ir: IR,
    env: Env[Any],
    args: IndexedSeq[(Any, Type)],
    aggArgs: Option[Agg],
    aggregator: Option[TypedAggregator[Any]],
    functionMemo: Memo[(PType, AsmFunction3[Region, Long, Boolean, Long])]): Any = {
    def interpret(ir: IR, env: Env[Any] = env, args: IndexedSeq[(Any, Type)] = args, aggArgs: Option[Agg] = aggArgs, aggregator: Option[TypedAggregator[Any]] = aggregator): Any =
      apply(ctx, ir, env, args, aggArgs, aggregator, functionMemo)

    ir match {
      case I32(x) => x
      case I64(x) => x
      case F32(x) => x
      case F64(x) => x
      case Str(x) => x
      case True() => true
      case False() => false
      case Literal(_, value) => value
      case Void() => ()
      case Cast(v, t) =>
        val vValue = interpret(v, env, args, aggArgs)
        if (vValue == null)
          null
        else
          (v.typ, t) match {
            case (_: TInt32, _: TInt32) => vValue
            case (_: TInt32, _: TInt64) => vValue.asInstanceOf[Int].toLong
            case (_: TInt32, _: TFloat32) => vValue.asInstanceOf[Int].toFloat
            case (_: TInt32, _: TFloat64) => vValue.asInstanceOf[Int].toDouble
            case (_: TInt64, _: TInt64) => vValue
            case (_: TInt64, _: TInt32) => vValue.asInstanceOf[Long].toInt
            case (_: TInt64, _: TFloat32) => vValue.asInstanceOf[Long].toFloat
            case (_: TInt64, _: TFloat64) => vValue.asInstanceOf[Long].toDouble
            case (_: TFloat32, _: TFloat32) => vValue
            case (_: TFloat32, _: TInt32) => vValue.asInstanceOf[Float].toInt
            case (_: TFloat32, _: TInt64) => vValue.asInstanceOf[Float].toLong
            case (_: TFloat32, _: TFloat64) => vValue.asInstanceOf[Float].toDouble
            case (_: TFloat64, _: TFloat64) => vValue
            case (_: TFloat64, _: TInt32) => vValue.asInstanceOf[Double].toInt
            case (_: TFloat64, _: TInt64) => vValue.asInstanceOf[Double].toLong
            case (_: TFloat64, _: TFloat32) => vValue.asInstanceOf[Double].toFloat
            case (_: TInt32, _: TCall) => vValue
          }
      case CastRename(v, _) => interpret(v)
      case NA(_) => null
      case IsNA(value) => interpret(value, env, args, aggArgs) == null
      case Coalesce(values) =>
        values.iterator
          .flatMap(x => Option(interpret(x, env, args, aggArgs)))
          .headOption
          .orNull
      case If(cond, cnsq, altr) =>
        assert(cnsq.typ == altr.typ)
        val condValue = interpret(cond, env, args, aggArgs)
        if (condValue == null)
          null
        else if (condValue.asInstanceOf[Boolean])
          interpret(cnsq, env, args, aggArgs)
        else
          interpret(altr, env, args, aggArgs)
      case Let(name, value, body) =>
        val valueValue = interpret(value, env, args, aggArgs)
        interpret(body, env.bind(name, valueValue), args, aggArgs)
      case Ref(name, _) => env.lookup(name)
      case ApplyBinaryPrimOp(op, l, r) =>
        val lValue = interpret(l, env, args, aggArgs)
        val rValue = interpret(r, env, args, aggArgs)
        if (lValue == null || rValue == null)
          null
        else
          (l.typ, r.typ) match {
            case (_: TInt32, _: TInt32) =>
              val ll = lValue.asInstanceOf[Int]
              val rr = rValue.asInstanceOf[Int]
              (op: @unchecked) match {
                case Add() => ll + rr
                case Subtract() => ll - rr
                case Multiply() => ll * rr
                case FloatingPointDivide() => ll.toFloat / rr.toFloat
                case RoundToNegInfDivide() => java.lang.Math.floorDiv(ll, rr)
                case BitAnd() => ll & rr
                case BitOr() => ll | rr
                case BitXOr() => ll ^ rr
                case LeftShift() => ll << rr
                case RightShift() => ll >> rr
                case LogicalRightShift() => ll >>> rr
              }
            case (_: TInt64, _: TInt32) =>
              val ll = lValue.asInstanceOf[Long]
              val rr = rValue.asInstanceOf[Int]
              (op: @unchecked) match {
                case LeftShift() => ll << rr
                case RightShift() => ll >> rr
                case LogicalRightShift() => ll >>> rr
              }
            case (_: TInt64, _: TInt64) =>
              val ll = lValue.asInstanceOf[Long]
              val rr = rValue.asInstanceOf[Long]
              (op: @unchecked) match {
                case Add() => ll + rr
                case Subtract() => ll - rr
                case Multiply() => ll * rr
                case FloatingPointDivide() => ll.toFloat / rr.toFloat
                case RoundToNegInfDivide() => java.lang.Math.floorDiv(ll, rr)
                case BitAnd() => ll & rr
                case BitOr() => ll | rr
                case BitXOr() => ll ^ rr
                case LeftShift() => ll << rr
                case RightShift() => ll >> rr
              }
            case (_: TFloat32, _: TFloat32) =>
              val ll = lValue.asInstanceOf[Float]
              val rr = rValue.asInstanceOf[Float]
              (op: @unchecked) match {
                case Add() => ll + rr
                case Subtract() => ll - rr
                case Multiply() => ll * rr
                case FloatingPointDivide() => ll / rr
                case RoundToNegInfDivide() => math.floor(ll / rr).toFloat
              }
            case (_: TFloat64, _: TFloat64) =>
              val ll = lValue.asInstanceOf[Double]
              val rr = rValue.asInstanceOf[Double]
              (op: @unchecked) match {
                case Add() => ll + rr
                case Subtract() => ll - rr
                case Multiply() => ll * rr
                case FloatingPointDivide() => ll / rr
                case RoundToNegInfDivide() => math.floor(ll / rr)
              }
          }
      case ApplyUnaryPrimOp(op, x) =>
        val xValue = interpret(x, env, args, aggArgs)
        if (xValue == null)
          null
        else op match {
          case Bang() =>
            assert(x.typ.isOfType(TBoolean()))
            !xValue.asInstanceOf[Boolean]
          case Negate() =>
            assert(x.typ.isInstanceOf[TNumeric])
            x.typ match {
              case TInt32(_) => -xValue.asInstanceOf[Int]
              case TInt64(_) => -xValue.asInstanceOf[Long]
              case TFloat32(_) => -xValue.asInstanceOf[Float]
              case TFloat64(_) => -xValue.asInstanceOf[Double]
            }
          case BitNot() =>
            x.typ match {
              case _: TInt32 => ~xValue.asInstanceOf[Int]
              case _: TInt64 => ~xValue.asInstanceOf[Long]
            }
        }
      case ApplyComparisonOp(op, l, r) =>
        val lValue = interpret(l, env, args, aggArgs)
        val rValue = interpret(r, env, args, aggArgs)
        if (op.strict && (lValue == null || rValue == null))
          null
        else
          op match {
            case EQ(t, _) => t.ordering.equiv(lValue, rValue)
            case EQWithNA(t, _) => t.ordering.equiv(lValue, rValue)
            case NEQ(t, _) => !t.ordering.equiv(lValue, rValue)
            case NEQWithNA(t, _) => !t.ordering.equiv(lValue, rValue)
            case LT(t, _) => t.ordering.lt(lValue, rValue)
            case GT(t, _) => t.ordering.gt(lValue, rValue)
            case LTEQ(t, _) => t.ordering.lteq(lValue, rValue)
            case GTEQ(t, _) => t.ordering.gteq(lValue, rValue)
            case Compare(t, _) => t.ordering.compare(lValue, rValue)
          }

      case MakeArray(elements, _) => elements.map(interpret(_, env, args, aggArgs)).toFastIndexedSeq
      case ArrayRef(a, i) =>
        val aValue = interpret(a, env, args, aggArgs)
        val iValue = interpret(i, env, args, aggArgs)
        if (aValue == null || iValue == null)
          null
        else {
          val a = aValue.asInstanceOf[IndexedSeq[Any]]
          val i = iValue.asInstanceOf[Int]
          if (i < 0 || i >= a.length)
            fatal(s"array index out of bounds: $i / ${ a.length }")
          else
            a.apply(i)
        }
      case ArrayLen(a) =>
        val aValue = interpret(a, env, args, aggArgs)
        if (aValue == null)
          null
        else
          aValue.asInstanceOf[IndexedSeq[Any]].length
      case ArrayRange(start, stop, step) =>
        val startValue = interpret(start, env, args, aggArgs)
        val stopValue = interpret(stop, env, args, aggArgs)
        val stepValue = interpret(step, env, args, aggArgs)
        if (stepValue == 0)
          fatal("Array range cannot have step size 0.")
        if (startValue == null || stopValue == null || stepValue == null)
          null
        else
          startValue.asInstanceOf[Int] until stopValue.asInstanceOf[Int] by stepValue.asInstanceOf[Int]
      case ArraySort(a, l, r, compare) =>
        val aValue = interpret(a, env, args, aggArgs)
        if (aValue == null)
          null
        else {
          aValue.asInstanceOf[IndexedSeq[Any]].sortWith { (left, right) =>
            if (left != null && right != null) {
              val res = interpret(compare, env.bind(l, left).bind(r, right), args, aggArgs)
              if (res == null)
                fatal("Result of sorting function cannot be missing.")
              res.asInstanceOf[Boolean]
            } else {
              right == null
            }
          }
        }
      case ToSet(a) =>
        val aValue = interpret(a, env, args, aggArgs)
        if (aValue == null)
          null
        else
          aValue.asInstanceOf[IndexedSeq[Any]].toSet
      case ToDict(a) =>
        val aValue = interpret(a, env, args, aggArgs)
        if (aValue == null)
          null
        else
          aValue.asInstanceOf[IndexedSeq[Row]].filter(_ != null).map { case Row(k, v) => (k, v) }.toMap

      case ToArray(c) =>
        val ordering = coerce[TIterable](c.typ).elementType.ordering.toOrdering
        val cValue = interpret(c, env, args, aggArgs)
        if (cValue == null)
          null
        else
          cValue match {
            case s: Set[_] =>
              s.asInstanceOf[Set[Any]].toFastIndexedSeq.sorted(ordering)
            case d: Map[_, _] => d.iterator.map { case (k, v) => Row(k, v) }.toFastIndexedSeq.sorted(ordering)
            case a => a
          }

      case LowerBoundOnOrderedCollection(orderedCollection, elem, onKey) =>
        val cValue = interpret(orderedCollection, env, args, aggArgs)
        val eValue = interpret(elem, env, args, aggArgs)
        if (cValue == null)
          null
        else {
          cValue match {
            case s: Set[_] =>
              assert(!onKey)
              s.count(elem.typ.ordering.lt(_, eValue))
            case d: Map[_, _] =>
              assert(onKey)
              d.count { case (k, _) => elem.typ.ordering.lt(k, eValue) }
            case a: IndexedSeq[_] =>
              if (onKey) {
                val (eltF, eltT) = orderedCollection.typ.asInstanceOf[TContainer].elementType match {
                  case t: TBaseStruct => ( { (x: Any) =>
                    val r = x.asInstanceOf[Row]
                    if (r == null) null else r.get(0)
                  }, t.types(0))
                  case i: TInterval => ( { (x: Any) =>
                    val i = x.asInstanceOf[Interval]
                    if (i == null) null else i.start
                  }, i.pointType)
                }
                val ordering = eltT.ordering
                val lb = a.count(elem => ordering.lt(eltF(elem), eValue))
                lb
              } else
                a.count(elem.typ.ordering.lt(_, eValue))
          }
        }

      case GroupByKey(collection) =>
        interpret(collection, env, args, aggArgs).asInstanceOf[IndexedSeq[Row]]
          .groupBy { case Row(k, _) => k }
          .mapValues { elt: IndexedSeq[Row] => elt.map { case Row(_, v) => v } }

      case ArrayMap(a, name, body) =>
        val aValue = interpret(a, env, args, aggArgs)
        if (aValue == null)
          null
        else {
          aValue.asInstanceOf[IndexedSeq[Any]].map { element =>
            interpret(body, env.bind(name, element), args, aggArgs)
          }
        }
      case ArrayFilter(a, name, cond) =>
        val aValue = interpret(a, env, args, aggArgs)
        if (aValue == null)
          null
        else {
          aValue.asInstanceOf[IndexedSeq[Any]].filter { element =>
            // casting to boolean treats null as false
            interpret(cond, env.bind(name, element), args, aggArgs).asInstanceOf[Boolean]
          }
        }
      case ArrayFlatMap(a, name, body) =>
        val aValue = interpret(a, env, args, aggArgs)
        if (aValue == null)
          null
        else {
          aValue.asInstanceOf[IndexedSeq[Any]].flatMap { element =>
            val r = interpret(body, env.bind(name, element), args, aggArgs).asInstanceOf[IndexedSeq[Any]]
            if (r != null)
              r
            else
              None
          }
        }
      case ArrayFold(a, zero, accumName, valueName, body) =>
        val aValue = interpret(a, env, args, aggArgs)
        if (aValue == null)
          null
        else {
          var zeroValue = interpret(zero, env, args, aggArgs)
          aValue.asInstanceOf[IndexedSeq[Any]].foreach { element =>
            zeroValue = interpret(body, env.bind(accumName -> zeroValue, valueName -> element), args, aggArgs)
          }
          zeroValue
        }
      case ArrayFold2(a, accum, valueName, seq, res) =>
        val aValue = interpret(a, env, args, aggArgs)
        if (aValue == null)
          null
        else {
          val accVals = accum.map { case (name, value) => (name, interpret(value, env, args, aggArgs)) }
          var e = env.bindIterable(accVals)
          aValue.asInstanceOf[IndexedSeq[Any]].foreach { elt =>
            e = e.bind(valueName, elt)
            accVals.indices.foreach { i =>
              e = e.bind(accum(i)._1, interpret(seq(i), e, args, aggArgs))
            }
          }
          interpret(res, e.delete(valueName), args, aggArgs)
        }
      case ArrayScan(a, zero, accumName, valueName, body) =>
        val aValue = interpret(a, env, args, aggArgs)
        if (aValue == null)
          null
        else {
          val zeroValue = interpret(zero, env, args, aggArgs)
          aValue.asInstanceOf[IndexedSeq[Any]].scanLeft(zeroValue) { (accum, elt) =>
            interpret(body, env.bind(accumName -> accum, valueName -> elt), args, aggArgs)
          }
        }

      case ArrayLeftJoinDistinct(left, right, l, r, compare, join) =>
        val lValue = interpret(left, env, args, aggArgs).asInstanceOf[IndexedSeq[Any]]
        val rValue = interpret(right, env, args, aggArgs).asInstanceOf[IndexedSeq[Any]].toIterator

        var relt: Any = if (rValue.hasNext) rValue.next() else null

        lValue.map { lelt =>
          while (rValue.hasNext && interpret(compare, env.bind(l -> lelt, r -> relt), args, aggArgs).asInstanceOf[Int] > 0) {
            relt = rValue.next()
          }
          if (interpret(compare, env.bind(l -> lelt, r -> relt), args, aggArgs).asInstanceOf[Int] == 0) {
            interpret(join, env.bind(l -> lelt, r -> relt), args, aggArgs)
          } else {
            interpret(join, env.bind(l -> lelt, r -> null), args, aggArgs)
          }
        }

      case ArrayFor(a, valueName, body) =>
        val aValue = interpret(a, env, args, aggArgs)
        if (aValue != null) {
          aValue.asInstanceOf[IndexedSeq[Any]].foreach { element =>
            interpret(body, env.bind(valueName -> element), args, aggArgs)
          }
        }
        ()

      case ArrayAgg(a, name, body) =>
        assert(aggArgs.isEmpty)
        val aggElementType = TStruct(name -> a.typ.asInstanceOf[TArray].elementType)
        val aValue = interpret(a, env, args, aggArgs)
          .asInstanceOf[IndexedSeq[Any]]
          .map(Row(_))
        interpret(body, env, args, Some(aValue -> aggElementType))

      case ArrayAggScan(a, name, query) =>
        throw new UnsupportedOperationException("ArrayAggScan")
      case Begin(xs) =>
        xs.foreach(x => interpret(x))
      case x@SeqOp(i, seqOpArgs, aggSig) =>
        assert(i == I32(0))
        aggSig.op match {
          case TakeBy() =>
            val IndexedSeq(a, ordering) = seqOpArgs
            aggregator.get.asInstanceOf[TakeByAggregator[_]].seqOp(interpret(a), interpret(ordering))
          case Count() =>
            assert(seqOpArgs.isEmpty)
            aggregator.get.asInstanceOf[CountAggregator].seqOp(0) // 0 is a dummy value
          case LinearRegression() =>
            val IndexedSeq(y, xs) = seqOpArgs
            aggregator.get.asInstanceOf[LinearRegressionAggregator].seqOp(interpret(y), interpret(xs))
          case Downsample() =>
            val IndexedSeq(x, y, label) = seqOpArgs
            aggregator.get.asInstanceOf[DownsampleAggregator].seqOp(interpret(x), interpret(y), interpret(label))
          case _ =>
            val IndexedSeq(a) = seqOpArgs
            aggregator.get.seqOp(interpret(a))
        }

      case x@AggFilter(cond, aggIR, isScan) =>
        assert(!isScan)
        // filters elements under aggregation environment
        val Some((aggElements, aggElementType)) = aggArgs
        val newAgg = aggElements.filter { row =>
          val env = (row.toSeq, aggElementType.fieldNames).zipped
            .foldLeft(Env.empty[Any]) { case (e, (v, n)) =>
              e.bind(n, v)
            }
          interpret(cond, env).asInstanceOf[Boolean]
        }
        interpret(aggIR, aggArgs = Some(newAgg -> aggElementType))

      case x@AggExplode(array, name, aggBody, isScan) =>
        assert(!isScan)
        // adds exploded array to elements under aggregation environment
        val Some((aggElements, aggElementType)) = aggArgs
        val newAggElementType = aggElementType.appendKey(name, coerce[TArray](array.typ).elementType)
        val newAgg = aggElements.flatMap { row =>
          val env = (row.toSeq, aggElementType.fieldNames).zipped
            .foldLeft(Env.empty[Any]) { case (e, (v, n)) =>
              e.bind(n, v)
            }
          interpret(array, env).asInstanceOf[IndexedSeq[Any]].map { elt =>
            Row(row.toSeq :+ elt: _*)
          }
        }
        interpret(aggBody, aggArgs = Some(newAgg -> newAggElementType))
      case x@AggGroupBy(key, aggIR, isScan) =>
        assert(!isScan)
        // evaluates one aggregation per key in aggregation environment
        val Some((aggElements, aggElementType)) = aggArgs
        val groupedAgg = aggElements.groupBy { row =>
          val env = (row.toSeq, aggElementType.fieldNames).zipped
            .foldLeft(Env.empty[Any]) { case (e, (v, n)) =>
              e.bind(n, v)
            }
          interpret(key, env)
        }
        groupedAgg.mapValues { row =>
          interpret(aggIR, aggArgs = Some((row, aggElementType)))
        }

      case x@AggArrayPerElement(a, elementName, indexName, aggBody, knownLength, isScan) => ???
      case x@ApplyAggOp(constructorArgs, initOpArgs, seqOpArgs, aggSig) =>
        assert(AggOp.getType(aggSig) == x.typ)

        def getAggregator(aggOp: AggOp, seqOpArgTypes: Seq[Type]): TypedAggregator[_] = (aggOp: @unchecked) match {
          case CallStats() =>
            assert(seqOpArgTypes == FastIndexedSeq(TCall()))
            val nAlleles = interpret(initOpArgs.get(0))
            new CallStatsAggregator(_ => nAlleles)
          case Count() => new CountAggregator()
          case Collect() =>
            val IndexedSeq(aggType) = seqOpArgTypes
            new CollectAggregator(aggType)
          case CollectAsSet() =>
            val IndexedSeq(aggType) = seqOpArgTypes
            new CollectSetAggregator(aggType)
          case Sum() =>
            val Seq(aggType) = seqOpArgTypes
            aggType match {
              case TInt64(_) => new SumAggregator[Long]()
              case TFloat64(_) => new SumAggregator[Double]()
              case TArray(TInt64(_), _) => new SumArrayAggregator[Long]()
              case TArray(TFloat64(_), _) => new SumArrayAggregator[Double]()
            }
          case Product() =>
            val Seq(aggType) = seqOpArgTypes
            aggType match {
              case TInt64(_) => new ProductAggregator[Long]()
              case TFloat64(_) => new ProductAggregator[Double]()
            }
          case Min() =>
            val Seq(aggType) = seqOpArgTypes
            aggType match {
              case TInt32(_) => new MinAggregator[Int, java.lang.Integer]()
              case TInt64(_) => new MinAggregator[Long, java.lang.Long]()
              case TFloat32(_) => new MinAggregator[Float, java.lang.Float]()
              case TFloat64(_) => new MinAggregator[Double, java.lang.Double]()
            }
          case Max() =>
            val Seq(aggType) = seqOpArgTypes
            aggType match {
              case TInt32(_) => new MaxAggregator[Int, java.lang.Integer]()
              case TInt64(_) => new MaxAggregator[Long, java.lang.Long]()
              case TFloat32(_) => new MaxAggregator[Float, java.lang.Float]()
              case TFloat64(_) => new MaxAggregator[Double, java.lang.Double]()
            }
          case Take() =>
            val Seq(n) = constructorArgs
            val Seq(aggType) = seqOpArgTypes
            val nValue = interpret(n, Env.empty[Any], null, null).asInstanceOf[Int]
            new TakeAggregator(aggType, nValue)
          case TakeBy() =>
            val Seq(n) = constructorArgs
            val Seq(aggType, _) = seqOpArgTypes
            val nValue = interpret(n, Env.empty[Any], null, null).asInstanceOf[Int]
            val ordering = seqOpArgs.last
            val ord = ordering.typ.ordering.toOrdering
            new TakeByAggregator(aggType, null, nValue)(ord)
          case LinearRegression() =>
            val Seq(k, k0) = constructorArgs
            val kValue = interpret(k, Env.empty[Any], null, null).asInstanceOf[Int]
            val k0Value = interpret(k0, Env.empty[Any], null, null).asInstanceOf[Int]
            val IndexedSeq(_, xType) = seqOpArgTypes
            new LinearRegressionAggregator(null, kValue, k0Value, xType)
          case Downsample() =>
            assert(seqOpArgTypes == FastIndexedSeq(TFloat64(), TFloat64(), TArray(TString())))
            val Seq(nDivisions) = constructorArgs
            val nDivisionsValue = interpret(nDivisions, Env.empty[Any], null, null).asInstanceOf[Int]
            new DownsampleAggregator(nDivisionsValue, null)
        }

        val aggregator = getAggregator(aggSig.op, aggSig.seqOpArgs)
        val Some((aggElements, aggElementType)) = aggArgs
        aggElements.foreach { element =>
          val env = (element.toSeq, aggElementType.fieldNames).zipped
            .foldLeft(Env.empty[Any]) { case (env, (v, n)) =>
              env.bind(n, v)
            }
          interpret(SeqOp(I32(0), seqOpArgs, aggSig), env, FastIndexedSeq(), None, Some(aggregator))
        }
        aggregator.result
      case x: ApplyScanOp =>
        throw new UnsupportedOperationException("interpreter doesn't support scans right now.")
      case MakeStruct(fields) =>
        Row.fromSeq(fields.map { case (name, fieldIR) => interpret(fieldIR, env, args, aggArgs) })
      case SelectFields(old, fields) =>
        val oldt = coerce[TStruct](old.typ)
        val oldRow = interpret(old, env, args, aggArgs).asInstanceOf[Row]
        if (oldRow == null)
          null
        else
          Row.fromSeq(fields.map(id => oldRow.get(oldt.fieldIdx(id))))
      case x@InsertFields(old, fields, fieldOrder) =>
        var struct = interpret(old, env, args, aggArgs)
        if (struct != null)
          fieldOrder match {
            case Some(fds) =>
              val newValues = fields.toMap.mapValues(interpret(_, env, args, aggArgs))
              val oldIndices = old.typ.asInstanceOf[TStruct].fields.map(f => f.name -> f.index).toMap
              Row.fromSeq(fds.map(name => newValues.getOrElse(name, struct.asInstanceOf[Row].get(oldIndices(name)))))
            case None =>
              var t = old.typ
              fields.foreach { case (name, body) =>
                val (newT, ins) = t.insert(body.typ, name)
                t = newT.asInstanceOf[TStruct]
                struct = ins(struct, interpret(body, env, args, aggArgs))
              }
              struct
          }
        else
          null
      case GetField(o, name) =>
        val oValue = interpret(o, env, args, aggArgs)
        if (oValue == null)
          null
        else {
          val oType = o.typ.asInstanceOf[TStruct]
          val fieldIndex = oType.fieldIdx(name)
          oValue.asInstanceOf[Row].get(fieldIndex)
        }
      case MakeTuple(types) =>
        Row.fromSeq(types.map { case (_, x) => interpret(x, env, args, aggArgs) })
      case GetTupleElement(o, idx) =>
        val oValue = interpret(o, env, args, aggArgs)
        if (oValue == null)
          null
        else
          oValue.asInstanceOf[Row].get(o.typ.asInstanceOf[TTuple].fieldIndex(idx))
      case In(i, _) =>
        val (a, _) = args(i)
        a
      case Die(message, typ) =>
        val message_ = interpret(message).asInstanceOf[String]
        fatal(if (message_ != null) message_ else "<exception message missing>")
      case ir@ApplyIR(function, functionArgs) =>
        interpret(ir.explicitNode, env, args, aggArgs)
      case ApplySpecial("||", Seq(left_, right_), _) =>
        val left = interpret(left_)
        if (left == true)
          true
        else {
          val right = interpret(right_)
          if (right == true)
            true
          else if (left == null || right == null)
            null
          else false
        }
      case ApplySpecial("&&", Seq(left_, right_), _) =>
        val left = interpret(left_)
        if (left == false)
          false
        else {
          val right = interpret(right_)
          if (right == false)
            false
          else if (left == null || right == null)
            null
          else true
        }
      case ir: AbstractApplyNode[_] =>
        val argTuple = PType.canonical(TTuple(ir.args.map(_.typ): _*)).asInstanceOf[PTuple]
        Region.scoped { region =>
          val (rt, f) = functionMemo.getOrElseUpdate(ir, {
            val wrappedArgs: IndexedSeq[BaseIR] = ir.args.zipWithIndex.map { case (x, i) =>
              GetTupleElement(Ref("in", argTuple.virtualType), i)
            }.toFastIndexedSeq
            val wrappedIR = Copy(ir, wrappedArgs)

            val (rt, makeFunction) = Compile[Long, Long](ctx, "in", argTuple, MakeTuple.ordered(FastSeq(wrappedIR)), optimize = false)
            (rt, makeFunction(0, region))
          })
          val rvb = new RegionValueBuilder()
          rvb.set(region)
          rvb.start(argTuple)
          rvb.startTuple()
          ir.args.zip(argTuple.types).foreach { case (arg, t) =>
            val argValue = interpret(arg, env, args, aggArgs)
            rvb.addAnnotation(t.virtualType, argValue)
          }
          rvb.endTuple()
          val offset = rvb.end()

          try {
            val resultOffset = f(region, offset, false)
            SafeRow(rt.asInstanceOf[PTuple], region, resultOffset).get(0)
          } catch {
            case e: Exception =>
              fatal(s"error while calling '${ ir.implementation.name }'", e)
          }
        }
      case Uniroot(functionid, fn, minIR, maxIR) =>
        val f = { x: Double => interpret(fn, env.bind(functionid, x), args, aggArgs).asInstanceOf[Double] }
        val min = interpret(minIR, env, args, aggArgs)
        val max = interpret(maxIR, env, args, aggArgs)
        if (min == null || max == null)
          null
        else
          stats.uniroot(f, min.asInstanceOf[Double], max.asInstanceOf[Double]).orNull
      case TableCount(child) =>
        child.partitionCounts
          .map(_.sum)
          .getOrElse(child.execute(ctx).rvd.count())
      case TableGetGlobals(child) =>
        child.execute(ctx).globals.safeJavaValue
      case TableCollect(child) =>
        val tv = child.execute(ctx)
        Row(tv.rvd.collect().toFastIndexedSeq, tv.globals.safeJavaValue)
      case TableMultiWrite(children, writer) =>
        val tvs = children.map(_.execute(ctx))
        writer(tvs)
      case TableWrite(child, writer) =>
        writer(child.execute(ctx))
      case BlockMatrixWrite(child, writer) =>
        val hc = HailContext.get
        writer(hc, child.execute(ctx))
      case BlockMatrixMultiWrite(blockMatrices, writer) =>
        writer(blockMatrices.map(_.execute(ctx)))
      case TableToValueApply(child, function) =>
        function.execute(ctx, child.execute(ctx))
      case BlockMatrixToValueApply(child, function) =>
        function.execute(ctx, child.execute(ctx))
      case TableAggregate(child, query) =>
        val value = child.execute(ctx)

        val globalsBc = value.globals.broadcast
        val globalsOffset = value.globals.value.offset

        val res = genUID()
        val extracted = agg.Extract(query, res)

        val wrapped = if (extracted.aggs.isEmpty) {
          val (rt: PTuple, f) = Compile[Long, Long](ctx,
            "global", value.globals.t,
            MakeTuple.ordered(FastSeq(extracted.postAggIR)))

          Region.scoped { region =>
            SafeRow(rt, region, f(0, region)(region, globalsOffset, false))
          }
        } else {
          val spec = BufferSpec.defaultUncompressed

          val (_, initOp) = CompileWithAggregators2[Long, Unit](ctx,
            extracted.aggs,
            "global", value.globals.t,
            extracted.init)

          val (_, partitionOpSeq) = CompileWithAggregators2[Long, Long, Unit](ctx,
            extracted.aggs,
            "global", value.globals.t,
            "row", value.rvd.rowPType,
            extracted.seqPerElt)

          val read = extracted.deserialize(ctx, spec)
          val write = extracted.serialize(ctx, spec)
          val combOpF = extracted.combOpF(ctx, spec)

          val (rTyp: PTuple, f) = CompileWithAggregators2[Long, Long](ctx,
            extracted.aggs,
            "global", value.globals.t,
            Let(res, extracted.results, MakeTuple.ordered(FastSeq(extracted.postAggIR))))
          assert(rTyp.types(0).virtualType == query.typ)

          val useTreeAggregate = extracted.shouldTreeAggregate
          val isCommutative = extracted.isCommutative
          log.info(s"Aggregate: useTreeAggregate=${ useTreeAggregate }")
          log.info(s"Aggregate: commutative=${ isCommutative }")

          val aggResults = value.rvd.combine[Array[Byte]](
            Region.scoped { region =>
              val initF = initOp(0, region)
              Region.scoped { aggRegion =>
                initF.newAggState(aggRegion)
                initF(region, globalsOffset, false)
                write(aggRegion, initF.getAggOffset())
              }
            },
            { (i: Int, ctx: RVDContext, it: Iterator[RegionValue]) =>
              val partRegion = ctx.freshRegion
              val globalsOffset = globalsBc.value.readRegionValue(partRegion)
              val init = initOp(i, partRegion)
              val seqOps = partitionOpSeq(i, partRegion)

              Region.smallScoped { aggRegion =>
                init.newAggState(aggRegion)
                init(partRegion, globalsOffset, false)
                seqOps.setAggState(aggRegion, init.getAggOffset())
                it.foreach { rv =>
                  seqOps(rv.region, globalsOffset, false, rv.offset, false)
                  ctx.region.clear()
                }
                write(aggRegion, seqOps.getAggOffset())
              }
            }, combOpF, commutative = isCommutative, tree = useTreeAggregate)

          Region.scoped { r =>
            val resF = f(0, r)
            Region.smallScoped { aggRegion =>
              resF.setAggState(aggRegion, read(aggRegion, aggResults))
              SafeRow(rTyp, r, resF(r, globalsOffset, false))
            }
          }
        }
        wrapped.get(0)
      case x: ReadPartition =>
        fatal(s"cannot interpret ${ Pretty(x) }")
    }
  }
}

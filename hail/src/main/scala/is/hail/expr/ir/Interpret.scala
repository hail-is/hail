package is.hail.expr.ir

import java.io.{PrintWriter, StringWriter}

import is.hail.{HailContext, Uploader, stats}
import is.hail.annotations.aggregators.RegionValueAggregator
import is.hail.annotations._
import is.hail.asm4s.AsmFunction3
import is.hail.expr.{JSONAnnotationImpex, TypedAggregator}
import is.hail.expr.types._
import is.hail.expr.types.physical.PTuple
import is.hail.expr.types.virtual._
import is.hail.methods._
import is.hail.utils._
import org.apache.spark.sql.Row
import org.json4s.JsonAST.JNull
import org.json4s.jackson.JsonMethods

import scala.collection.JavaConverters._

object Interpret {
  type Agg = (IndexedSeq[Row], TStruct)

  def interpretPyIR(s: String, refMap: java.util.HashMap[String, Type], irMap: java.util.HashMap[String, BaseIR]): String = {
    interpretPyIR(s, refMap.asScala.toMap, irMap.asScala.toMap)
  }

  def interpretPyIR(s: String, refMap: Map[String, Type] = Map.empty, irMap: Map[String, BaseIR] = Map.empty): String = {
    val ir = IRParser.parse_value_ir(s, IRParserEnvironment(refMap, irMap))
    val t = ir.typ
    val value = Interpret[Any](ir)

    val jsonValue = t match {
      case TVoid => JNull
      case _ => JSONAnnotationImpex.exportAnnotation(value, t)
    }

    JsonMethods.compact(jsonValue)
  }

  def apply(tir: TableIR): TableValue =
    apply(tir, optimize = true)

  def apply(tir: TableIR, optimize: Boolean): TableValue = {
    val tiropt = if (optimize)
      Optimize(tir)
    else
      tir

    val lowered = LowerMatrixIR(LiftLiterals(tiropt).asInstanceOf[TableIR])
    val lowopt = if (optimize)
      Optimize(lowered, noisy = true, canGenerateLiterals = false)
    else
      lowered

    lowopt.execute(HailContext.get)
  }

  def apply(mir: MatrixIR): MatrixValue =
    apply(mir, optimize = true)

  def apply(mir: MatrixIR, optimize: Boolean): MatrixValue = {
    val miropt = if (optimize)
      Optimize(mir)
    else
      mir

    val lowered = LowerMatrixIR(LiftLiterals(miropt).asInstanceOf[MatrixIR])
    val lowopt = if (optimize)
      Optimize(lowered, noisy = true, canGenerateLiterals = false)
    else
      lowered

    lowopt.execute(HailContext.get)
  }

  def apply[T](ir: IR): T = apply(ir, Env.empty[(Any, Type)], FastIndexedSeq(), None).asInstanceOf[T]

  def apply[T](ir: IR, optimize: Boolean): T = apply(ir, Env.empty[(Any, Type)], FastIndexedSeq(), None, optimize).asInstanceOf[T]

  def apply[T](ir0: IR,
    env: Env[(Any, Type)],
    args: IndexedSeq[(Any, Type)],
    agg: Option[Agg],
    optimize: Boolean = true
  ): T = {
    val (typeEnv, valueEnv) = env.m.foldLeft((Env.empty[Type], Env.empty[Any])) {
      case ((e1, e2), (k, (value, t))) => (e1.bind(k, t), e2.bind(k, value))
    }

    var ir = ir0.unwrap

    def optimizeIR(canGenerateLiterals: Boolean) {
      ir = Optimize(ir, noisy = true, canGenerateLiterals)
      TypeCheck(ir, typeEnv, agg.map { agg =>
        agg._2.fields.foldLeft(Env.empty[Type]) { case (env, f) =>
          env.bind(f.name, f.typ)
        }
      })
    }

    if (optimize) optimizeIR(true)
    ir = LiftLiterals(ir).asInstanceOf[IR]
    ir = LowerMatrixIR(ir)
    if (optimize) optimizeIR(false)

    val result = apply(ir, valueEnv, args, agg, None, Memo.empty[AsmFunction3[Region, Long, Boolean, Long]]).asInstanceOf[T]

    Uploader.uploadPipeline(ir0, ir)

    result
  }

  private def apply(ir: IR, env: Env[Any], args: IndexedSeq[(Any, Type)], agg: Option[Agg], aggregator: Option[TypedAggregator[Any]], functionMemo: Memo[AsmFunction3[Region, Long, Boolean, Long]]): Any = {
    def interpret(ir: IR, env: Env[Any] = env, args: IndexedSeq[(Any, Type)] = args, agg: Option[Agg] = agg, aggregator: Option[TypedAggregator[Any]] = aggregator): Any =
      apply(ir, env, args, agg, aggregator, functionMemo)

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
        val vValue = interpret(v, env, args, agg)
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
      case NA(_) => null
      case IsNA(value) => interpret(value, env, args, agg) == null
      case If(cond, cnsq, altr) =>
        assert(cnsq.typ == altr.typ)
        val condValue = interpret(cond, env, args, agg)
        if (condValue == null)
          null
        else if (condValue.asInstanceOf[Boolean])
          interpret(cnsq, env, args, agg)
        else
          interpret(altr, env, args, agg)
      case Let(name, value, body) =>
        val valueValue = interpret(value, env, args, agg)
        interpret(body, env.bind(name, valueValue), args, agg)
      case Ref(name, _) => env.lookup(name)
      case ApplyBinaryPrimOp(op, l, r) =>
        val lValue = interpret(l, env, args, agg)
        val rValue = interpret(r, env, args, agg)
        if (lValue == null || rValue == null)
          null
        else
          (l.typ, r.typ) match {
            case (_: TInt32, _: TInt32) =>
              val ll = lValue.asInstanceOf[Int]
              val rr = rValue.asInstanceOf[Int]
              op match {
                case Add() => ll + rr
                case Subtract() => ll - rr
                case Multiply() => ll * rr
                case FloatingPointDivide() => ll.toFloat / rr.toFloat
                case RoundToNegInfDivide() => java.lang.Math.floorDiv(ll, rr)
              }
            case (_: TInt64, _: TInt64) =>
              val ll = lValue.asInstanceOf[Long]
              val rr = rValue.asInstanceOf[Long]
              op match {
                case Add() => ll + rr
                case Subtract() => ll - rr
                case Multiply() => ll * rr
                case FloatingPointDivide() => ll.toFloat / rr.toFloat
                case RoundToNegInfDivide() => java.lang.Math.floorDiv(ll, rr)
              }
            case (_: TFloat32, _: TFloat32) =>
              val ll = lValue.asInstanceOf[Float]
              val rr = rValue.asInstanceOf[Float]
              op match {
                case Add() => ll + rr
                case Subtract() => ll - rr
                case Multiply() => ll * rr
                case FloatingPointDivide() => ll / rr
                case RoundToNegInfDivide() => math.floor(ll / rr).toFloat
              }
            case (_: TFloat64, _: TFloat64) =>
              val ll = lValue.asInstanceOf[Double]
              val rr = rValue.asInstanceOf[Double]
              op match {
                case Add() => ll + rr
                case Subtract() => ll - rr
                case Multiply() => ll * rr
                case FloatingPointDivide() => ll / rr
                case RoundToNegInfDivide() => math.floor(ll / rr)
              }
          }
      case ApplyUnaryPrimOp(op, x) =>
        op match {
          case Bang() =>
            assert(x.typ.isOfType(TBoolean()))
            val xValue = interpret(x, env, args, agg)
            if (xValue == null)
              null
            else
              !xValue.asInstanceOf[Boolean]
          case Negate() =>
            assert(x.typ.isInstanceOf[TNumeric])
            val xValue = interpret(x, env, args, agg)
            if (xValue == null)
              null
            else {
              x.typ match {
                case TInt32(_) => -xValue.asInstanceOf[Int]
                case TInt64(_) => -xValue.asInstanceOf[Long]
                case TFloat32(_) => -xValue.asInstanceOf[Float]
                case TFloat64(_) => -xValue.asInstanceOf[Double]
              }
            }
        }
      case ApplyComparisonOp(op, l, r) =>
        val lValue = interpret(l, env, args, agg)
        val rValue = interpret(r, env, args, agg)
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
          }

      case MakeArray(elements, _) => elements.map(interpret(_, env, args, agg)).toFastIndexedSeq
      case ArrayRef(a, i) =>
        val aValue = interpret(a, env, args, agg)
        val iValue = interpret(i, env, args, agg)
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
        val aValue = interpret(a, env, args, agg)
        if (aValue == null)
          null
        else
          aValue.asInstanceOf[IndexedSeq[Any]].length
      case ArrayRange(start, stop, step) =>
        val startValue = interpret(start, env, args, agg)
        val stopValue = interpret(stop, env, args, agg)
        val stepValue = interpret(step, env, args, agg)
        if (stepValue == 0)
          fatal("Array range cannot have step size 0.")
        if (startValue == null || stopValue == null || stepValue == null)
          null
        else
          startValue.asInstanceOf[Int] until stopValue.asInstanceOf[Int] by stepValue.asInstanceOf[Int]
      case ArraySort(a, ascending, onKey) =>
        val aValue = interpret(a, env, args, agg)
        val ascendingValue = interpret(ascending, env, args, agg)
        if (aValue == null)
          null
        else {
          var sortType = a.typ.asInstanceOf[TArray].elementType
          if (onKey)
            sortType = sortType.asInstanceOf[TBaseStruct].types(0)
          val ord =
            if (ascendingValue == null || ascendingValue.asInstanceOf[Boolean])
              sortType.ordering
            else
              sortType.ordering.reverse
          if (onKey)
            aValue.asInstanceOf[IndexedSeq[Row]].sortBy(_.get(0))(ord.toOrdering)
          else
            aValue.asInstanceOf[IndexedSeq[Any]].sorted(ord.toOrdering)
        }
      case ToSet(a) =>
        val aValue = interpret(a, env, args, agg)
        if (aValue == null)
          null
        else
          aValue.asInstanceOf[IndexedSeq[Any]].toSet
      case ToDict(a) =>
        val aValue = interpret(a, env, args, agg)
        if (aValue == null)
          null
        else
          aValue.asInstanceOf[IndexedSeq[Row]].filter(_ != null).map { case Row(k, v) => (k, v) }.toMap

      case ToArray(c) =>
        val ordering = coerce[TContainer](c.typ).elementType.ordering.toOrdering
        val cValue = interpret(c, env, args, agg)
        if (cValue == null)
          null
        else
          cValue match {
            case s: Set[Any] =>
              s.toFastIndexedSeq.sorted(ordering)
            case d: Map[_, _] => d.iterator.map { case (k, v) => Row(k, v) }.toFastIndexedSeq.sorted(ordering)
            case a => a
          }

      case LowerBoundOnOrderedCollection(orderedCollection, elem, onKey) =>
        val cValue = interpret(orderedCollection, env, args, agg)
        val eValue = interpret(elem, env, args, agg)
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
              assert(!onKey)
              a.count(elem.typ.ordering.lt(_, eValue))
          }
        }

      case GroupByKey(collection) =>
        interpret(collection, env, args, agg).asInstanceOf[IndexedSeq[Row]]
          .groupBy { case Row(k, _) => k }
          .mapValues { elt: IndexedSeq[Row] => elt.map { case Row(_, v) => v } }

      case ArrayMap(a, name, body) =>
        val aValue = interpret(a, env, args, agg)
        if (aValue == null)
          null
        else {
          aValue.asInstanceOf[IndexedSeq[Any]].map { element =>
            interpret(body, env.bind(name, element), args, agg)
          }
        }
      case ArrayFilter(a, name, cond) =>
        val aValue = interpret(a, env, args, agg)
        if (aValue == null)
          null
        else {
          aValue.asInstanceOf[IndexedSeq[Any]].filter { element =>
            // casting to boolean treats null as false
            interpret(cond, env.bind(name, element), args, agg).asInstanceOf[Boolean]
          }
        }
      case ArrayFlatMap(a, name, body) =>
        val aValue = interpret(a, env, args, agg)
        if (aValue == null)
          null
        else {
          aValue.asInstanceOf[IndexedSeq[Any]].flatMap { element =>
            val r = interpret(body, env.bind(name, element), args, agg).asInstanceOf[IndexedSeq[Any]]
            if (r != null)
              r
            else
              None
          }
        }
      case ArrayFold(a, zero, accumName, valueName, body) =>
        val aValue = interpret(a, env, args, agg)
        if (aValue == null)
          null
        else {
          var zeroValue = interpret(zero, env, args, agg)
          aValue.asInstanceOf[IndexedSeq[Any]].foreach { element =>
            zeroValue = interpret(body, env.bind(accumName -> zeroValue, valueName -> element), args, agg)
          }
          zeroValue
        }
      case ArrayScan(a, zero, accumName, valueName, body) =>
        val aValue = interpret(a, env, args, agg)
        if (aValue == null)
          null
        else {
          val zeroValue = interpret(zero, env, args, agg)
          aValue.asInstanceOf[IndexedSeq[Any]].scanLeft(zeroValue) { (accum, elt) =>
            interpret(body, env.bind(accumName -> accum, valueName -> elt), args, agg)
          }
        }

      case ArrayFor(a, valueName, body) =>
        val aValue = interpret(a, env, args, agg)
        if (aValue != null) {
          aValue.asInstanceOf[IndexedSeq[Any]].foreach { element =>
            interpret(body, env.bind(valueName -> element), args, agg)
          }
        }
        ()
      case Begin(xs) =>
        xs.foreach(x => Interpret(x))
      case x@SeqOp(i, seqOpArgs, aggSig) =>
        assert(i == I32(0))
        aggSig.op match {
          case Inbreeding() =>
            val IndexedSeq(a, af) = seqOpArgs
            aggregator.get.asInstanceOf[InbreedingAggregator].seqOp(interpret(a), interpret(af))
          case TakeBy() =>
            val IndexedSeq(a, ordering) = seqOpArgs
            aggregator.get.asInstanceOf[TakeByAggregator[_]].seqOp(interpret(a), interpret(ordering))
          case Count() =>
            assert(seqOpArgs.isEmpty)
            aggregator.get.asInstanceOf[CountAggregator].seqOp(0) // 0 is a dummy value
          case LinearRegression() =>
            val IndexedSeq(y, xs) = seqOpArgs
            aggregator.get.asInstanceOf[LinearRegressionAggregator].seqOp(interpret(y), interpret(xs))
          case PearsonCorrelation() =>
            val IndexedSeq(x, y) = seqOpArgs
            aggregator.get.asInstanceOf[PearsonCorrelationAggregator].seqOp((interpret(x), interpret(y)))
          case Downsample() =>
            val IndexedSeq(x, y, label) = seqOpArgs
            aggregator.get.asInstanceOf[DownsampleAggregator].seqOp(interpret(x), interpret(y), interpret(label))
          case _ =>
            val IndexedSeq(a) = seqOpArgs
            aggregator.get.seqOp(interpret(a))
        }

      case x@AggFilter(cond, aggIR) =>
        // filters elements under aggregation environment
        val Some((aggElements, aggElementType)) = agg
        val newAgg = aggElements.filter { row =>
          val env = (row.toSeq, aggElementType.fieldNames).zipped
            .foldLeft(Env.empty[Any]) { case (e, (v, n)) =>
              e.bind(n, v)
            }
          interpret(cond, env).asInstanceOf[Boolean]
        }
        interpret(aggIR, agg = Some(newAgg -> aggElementType))

      case x@AggExplode(array, name, aggBody) =>
        // adds exploded array to elements under aggregation environment
        val Some((aggElements, aggElementType)) = agg
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
        interpret(aggBody, agg = Some(newAgg -> newAggElementType))
      case x@AggGroupBy(key, aggIR) =>
        // evaluates one aggregation per key in aggregation environment
        val Some((aggElements, aggElementType)) = agg
        val groupedAgg = aggElements.groupBy { row =>
          val env = (row.toSeq, aggElementType.fieldNames).zipped
            .foldLeft(Env.empty[Any]) { case (e, (v, n)) =>
              e.bind(n, v)
            }
          interpret(key, env)
        }
        groupedAgg.mapValues { row =>
          interpret(aggIR, agg=Some(row, aggElementType))
        }

      case x@ApplyAggOp(constructorArgs, initOpArgs, seqOpArgs, aggSig) =>
        assert(AggOp.getType(aggSig) == x.typ)

        def getAggregator(aggOp: AggOp, seqOpArgTypes: Seq[Type]): TypedAggregator[_] = aggOp match {
          case CallStats() =>
            assert(seqOpArgTypes == FastIndexedSeq(TCall()))
            val nAlleles = interpret(initOpArgs.get(0))
            new CallStatsAggregator(_ => nAlleles)
          case Inbreeding() =>
            assert(seqOpArgTypes == FastIndexedSeq(TCall(), TFloat64()))
            new InbreedingAggregator(null)
          case HardyWeinberg() =>
            assert(seqOpArgTypes == FastIndexedSeq(TCall()))
            new HWEAggregator()
          case Count() => new CountAggregator()
          case Collect() =>
            val IndexedSeq(aggType) = seqOpArgTypes
            new CollectAggregator(aggType)
          case Counter() =>
            val IndexedSeq(aggType) = seqOpArgTypes
            new CounterAggregator(aggType)
          case CollectAsSet() =>
            val IndexedSeq(aggType) = seqOpArgTypes
            new CollectSetAggregator(aggType)
          case Fraction() =>
            assert(seqOpArgTypes == FastIndexedSeq(TBoolean()))
            new FractionAggregator(a => a)
          case Sum() =>
            val IndexedSeq(aggType) = seqOpArgTypes
            aggType match {
              case TInt64(_) => new SumAggregator[Long]()
              case TFloat64(_) => new SumAggregator[Double]()
              case TArray(TInt64(_), _) => new SumArrayAggregator[Long]()
              case TArray(TFloat64(_), _) => new SumArrayAggregator[Double]()
            }
          case Product() =>
            val IndexedSeq(aggType) = seqOpArgTypes
            aggType match {
              case TInt64(_) => new ProductAggregator[Long]()
              case TFloat64(_) => new ProductAggregator[Double]()
            }
          case Min() =>
            val IndexedSeq(aggType) = seqOpArgTypes
            aggType match {
              case TInt32(_) => new MinAggregator[Int, java.lang.Integer]()
              case TInt64(_) => new MinAggregator[Long, java.lang.Long]()
              case TFloat32(_) => new MinAggregator[Float, java.lang.Float]()
              case TFloat64(_) => new MinAggregator[Double, java.lang.Double]()
            }
          case Max() =>
            val IndexedSeq(aggType) = seqOpArgTypes
            aggType match {
              case TInt32(_) => new MaxAggregator[Int, java.lang.Integer]()
              case TInt64(_) => new MaxAggregator[Long, java.lang.Long]()
              case TFloat32(_) => new MaxAggregator[Float, java.lang.Float]()
              case TFloat64(_) => new MaxAggregator[Double, java.lang.Double]()
            }
          case Take() =>
            val IndexedSeq(n) = constructorArgs
            val IndexedSeq(aggType) = seqOpArgTypes
            val nValue = interpret(n, Env.empty[Any], null, null).asInstanceOf[Int]
            new TakeAggregator(aggType, nValue)
          case TakeBy() =>
            val IndexedSeq(n) = constructorArgs
            val IndexedSeq(aggType, _) = seqOpArgTypes
            val nValue = interpret(n, Env.empty[Any], null, null).asInstanceOf[Int]
            val ordering = seqOpArgs.last
            val ord = ordering.typ.ordering.toOrdering
            new TakeByAggregator(aggType, null, nValue)(ord)
          case Statistics() => new StatAggregator()
          case InfoScore() =>
            val IndexedSeq(aggType) = seqOpArgTypes
            new InfoScoreAggregator(aggType.physicalType)
          case Histogram() =>
            val Seq(start, end, bins) = constructorArgs
            val startValue = interpret(start, Env.empty[Any], null, null).asInstanceOf[Double]
            val endValue = interpret(end, Env.empty[Any], null, null).asInstanceOf[Double]
            val binsValue = interpret(bins, Env.empty[Any], null, null).asInstanceOf[Int]

            if (binsValue <= 0)
              fatal(s"""method `hist' expects `bins' argument to be > 0, but got $bins""")

            val binSize = (endValue - startValue) / binsValue
            if (binSize <= 0)
              fatal(
                s"""invalid bin size from given arguments (start = $startValue, end = $endValue, bins = $binsValue)
                   |  Method requires positive bin size [(end - start) / bins], but got ${ binSize.formatted("%.2f") }
                  """.stripMargin)

            val indices = Array.tabulate(binsValue + 1)(i => startValue + i * binSize)
            new HistAggregator(indices)
          case PearsonCorrelation() => new PearsonCorrelationAggregator()
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
        val Some((aggElements, aggElementType)) = agg
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
        Row.fromSeq(fields.map { case (name, fieldIR) => interpret(fieldIR, env, args, agg) })
      case SelectFields(old, fields) =>
        val oldt = coerce[TStruct](old.typ)
        val oldRow = interpret(old, env, args, agg).asInstanceOf[Row]
        if (oldRow == null)
          null
        else
          Row.fromSeq(fields.map(id => oldRow.get(oldt.fieldIdx(id))))
      case InsertFields(old, fields) =>
        var struct = interpret(old, env, args, agg)
        var t = old.typ
        if (struct != null)
          fields.foreach { case (name, body) =>
            val (newT, ins) = t.insert(body.typ, name)
            t = newT.asInstanceOf[TStruct]
            struct = ins(struct, interpret(body, env, args, agg))
          }
        struct
      case GetField(o, name) =>
        val oValue = interpret(o, env, args, agg)
        if (oValue == null)
          null
        else {
          val oType = o.typ.asInstanceOf[TStruct]
          val fieldIndex = oType.fieldIdx(name)
          oValue.asInstanceOf[Row].get(fieldIndex)
        }
      case MakeTuple(types) =>
        Row.fromSeq(types.map(x => interpret(x, env, args, agg)))
      case GetTupleElement(o, idx) =>
        val oValue = interpret(o, env, args, agg)
        if (oValue == null)
          null
        else
          oValue.asInstanceOf[Row].get(idx)
      case StringSlice(s, start, end) =>
        val Array(maybeString, vstart: Int, vend: Int) =
          Array(s, start, end).map(interpret(_, env, args, agg))
        if (maybeString == null)
          null
        else {
          val vs = maybeString.asInstanceOf[String]
          val length = vs.length
          var start_ = if (vstart < 0) vstart + length else vstart
          start_ = math.min(math.max(start_, 0), length)
          var end_ = if (vend < 0) vend + length else vend
          end_ = math.min(math.max(end_, start_), length)
          vs.substring(start_, end_)
        }
      case StringLength(s) =>
        val vs = interpret(s).asInstanceOf[String]
        if (vs == null) null else vs.getBytes().length
      case In(i, _) =>
        val (a, _) = args(i)
        a
      case Die(message, typ) =>
        val message_ = interpret(message).asInstanceOf[String]
        fatal(if (message_ != null) message_ else "<exception message missing>")
      case ir@ApplyIR(function, functionArgs, conversion) =>
        interpret(ir.explicitNode, env, args, agg)
      case ApplySpecial("||", Seq(left_, right_)) =>
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
      case ApplySpecial("&&", Seq(left_, right_)) =>
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
        val argTuple = TTuple(ir.args.map(_.typ): _*).physicalType
        val f = functionMemo.getOrElseUpdate(ir, {
          val wrappedArgs: IndexedSeq[BaseIR] = ir.args.zipWithIndex.map { case (x, i) =>
            GetTupleElement(Ref("in", argTuple.virtualType                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          ), i)
          }.toFastIndexedSeq
          val wrappedIR = Copy(ir, wrappedArgs).asInstanceOf[IR]

          val (_, makeFunction) = Compile[Long, Long]("in", argTuple, MakeTuple(List(wrappedIR)))
          makeFunction(0)
        })
        Region.scoped { region =>
          val rvb = new RegionValueBuilder()
          rvb.set(region)
          rvb.start(argTuple)
          rvb.startTuple()
          ir.args.zip(argTuple.types).foreach { case (arg, t) =>
            val argValue = interpret(arg, env, args, agg)
            rvb.addAnnotation(t.virtualType, argValue)
          }
          rvb.endTuple()
          val offset = rvb.end()

          val resultOffset = f(region, offset, false)
          SafeRow(PTuple(FastIndexedSeq(ir.implementation.returnType.subst().physicalType), required = true), region, resultOffset)
            .get(0)
        }
      case Uniroot(functionid, fn, minIR, maxIR) =>
        val f = { x: Double => interpret(fn, env.bind(functionid, x), args, agg).asInstanceOf[Double] }
        val min = interpret(minIR, env, args, agg)
        val max = interpret(maxIR, env, args, agg)
        if (min == null || max == null)
          null
        else
          stats.uniroot(f, min.asInstanceOf[Double], max.asInstanceOf[Double]).orNull

      case TableCount(child) =>
        child.partitionCounts
          .map(_.sum)
          .getOrElse(child.execute(HailContext.get).rvd.count())
      case TableGetGlobals(child) =>
        child.execute(HailContext.get).globals.value
      case MatrixWrite(child, writer) =>
        val mv = child.execute(HailContext.get)
        writer(mv)
      case TableWrite(child, path, overwrite, stageLocally, codecSpecJSONStr) =>
        val hc = HailContext.get
        val tableValue = child.execute(hc)
        tableValue.write(path, overwrite, stageLocally, codecSpecJSONStr)
      case TableExport(child, path, typesFile, header, exportType) =>
        val hc = HailContext.get
        val tableValue = child.execute(hc)
        tableValue.export(path, typesFile, header, exportType)
      case TableAggregate(child, query) =>
        val localGlobalSignature = child.typ.globalType
        val (rvAggs, initOps, seqOps, aggResultType, postAggIR) = CompileWithAggregators[Long, Long, Long](
          "global", child.typ.globalType.physicalType,
          "global", child.typ.globalType.physicalType,
          "row", child.typ.rowType.physicalType,
          MakeTuple(Array(query)), "AGGR",
          (nAggs: Int, initOpIR: IR) => initOpIR,
          (nAggs: Int, seqOpIR: IR) => seqOpIR)

        val (t, f) = Compile[Long, Long, Long](
          "AGGR", aggResultType,
          "global", child.typ.globalType.physicalType,
          postAggIR)

        val value = child.execute(HailContext.get)
        val globalsBc = value.globals.broadcast

        val aggResults = if (rvAggs.nonEmpty) {
          Region.scoped { region =>
            val rvb: RegionValueBuilder = new RegionValueBuilder()
            rvb.set(region)

            rvb.start(localGlobalSignature.physicalType)
            rvb.addAnnotation(localGlobalSignature, globalsBc.value)
            val globals = rvb.end()

            initOps(0)(region, rvAggs, globals, false)
          }

          val combOp = { (rvAggs1: Array[RegionValueAggregator], rvAggs2: Array[RegionValueAggregator]) =>
            rvAggs1.zip(rvAggs2).foreach { case (rvAgg1, rvAgg2) => rvAgg1.combOp(rvAgg2) }
            rvAggs1
          }

          value.rvd.aggregateWithPartitionOp(rvAggs, (i, ctx) => {
            val r = ctx.freshRegion
            val rvb = new RegionValueBuilder()
            rvb.set(r)
            rvb.start(localGlobalSignature.physicalType)
            rvb.addAnnotation(localGlobalSignature, globalsBc.value)
            val globalsOffset = rvb.end()
            val seqOpsFunction = seqOps(i)
            (globalsOffset, seqOpsFunction)
          })({ case ((globalsOffset, seqOpsFunction), comb, rv) =>
            seqOpsFunction(rv.region, comb, globalsOffset, false, rv.offset, false)
          }, combOp)
        } else
          Array.empty[RegionValueAggregator]

        Region.scoped { region =>
          val rvb: RegionValueBuilder = new RegionValueBuilder()
          rvb.set(region)

          rvb.start(aggResultType)
          rvb.startTuple()
          aggResults.foreach(_.result(rvb))
          rvb.endTuple()
          val aggResultsOffset = rvb.end()

          rvb.start(localGlobalSignature.physicalType)
          rvb.addAnnotation(localGlobalSignature, globalsBc.value)
          val globalsOffset = rvb.end()

          val resultOffset = f(0)(region, aggResultsOffset, false, globalsOffset, false)

          SafeRow(coerce[PTuple](t), region, resultOffset)
            .get(0)
        }
      case MatrixAggregate(child, query) =>
        val localGlobalSignature = child.typ.globalType
        val value = child.execute(HailContext.get)
        val colArrayType = TArray(child.typ.colType)
        val (rvAggs, initOps, seqOps, aggResultType, postAggIR) = CompileWithAggregators[Long, Long, Long, Long](
          "global", child.typ.globalType.physicalType,
          "global", child.typ.globalType.physicalType,
          "sa", colArrayType.physicalType,
          "va", child.typ.rvRowType.physicalType,
          MakeTuple(Array(query)), "AGGR",
          (nAggs: Int, initOpIR: IR) => initOpIR,
          (nAggs: Int, seqOpIR: IR) =>
            ArrayFor(
              ArrayRange(I32(0), I32(value.nCols), I32(1)),
              "idx",
              Let(
                "sa",
                ArrayRef(Ref("sa", colArrayType), Ref("idx", TInt32Optional)),
                Let(
                  "g",
                  ArrayRef(GetField(Ref("va", child.typ.rvRowType), MatrixType.entriesIdentifier), Ref("idx", TInt32Optional)),
                  seqOpIR))))

        val (t, f) = Compile[Long, Long, Long](
          "AGGR", aggResultType,
          "global", child.typ.globalType.physicalType,
          postAggIR)

        val globalsBc = value.globals.broadcast
        val colValuesBc = value.colValues.broadcast
        val localColsType = TArray(child.typ.colType)

        val aggResults = if (rvAggs.nonEmpty) {
          Region.scoped { region =>
            val rvb: RegionValueBuilder = new RegionValueBuilder()
            rvb.set(region)

            rvb.start(localGlobalSignature.physicalType)
            rvb.addAnnotation(localGlobalSignature, globalsBc.value)
            val globals = rvb.end()

            initOps(0)(region, rvAggs, globals, false)
          }

          val zval = rvAggs
          val combOp = { (rvAggs1: Array[RegionValueAggregator], rvAggs2: Array[RegionValueAggregator]) =>
            rvAggs1.zip(rvAggs2).foreach { case (rvAgg1, rvAgg2) => rvAgg1.combOp(rvAgg2) }
            rvAggs1
          }

          value.rvd.aggregateWithPartitionOp(rvAggs, (i, ctx) => {
            val r = ctx.freshRegion
            val rvb = new RegionValueBuilder()
            rvb.set(r)
            rvb.start(localGlobalSignature.physicalType)
            rvb.addAnnotation(localGlobalSignature, globalsBc.value)
            val globalsOffset = rvb.end()
            rvb.start(localColsType.physicalType)
            rvb.addAnnotation(localColsType, colValuesBc.value)
            val colsOffset = rvb.end()
            val seqOpsFunction = seqOps(i)
            (globalsOffset, colsOffset, seqOpsFunction)
          })({ case ((globalsOffset, colsOffset, seqOpsFunction), comb, rv) =>
            seqOpsFunction(rv.region, comb, globalsOffset, false, colsOffset, false, rv.offset, false)
          }, combOp)
        } else
          Array.empty[RegionValueAggregator]

        Region.scoped { region =>
          val rvb: RegionValueBuilder = new RegionValueBuilder()
          rvb.set(region)

          rvb.start(aggResultType)
          rvb.startTuple()
          aggResults.foreach(_.result(rvb))
          rvb.endTuple()
          val aggResultsOffset = rvb.end()

          rvb.start(localGlobalSignature.physicalType)
          rvb.addAnnotation(localGlobalSignature, globalsBc.value)
          val globalsOffset = rvb.end()

          val resultOffset = f(0)(region, aggResultsOffset, false, globalsOffset, false)

          SafeRow(coerce[PTuple](t), region, resultOffset)
            .get(0)
        }
    }
  }
}

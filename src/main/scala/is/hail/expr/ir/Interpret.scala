package is.hail.expr.ir

import is.hail.{HailContext, stats}
import is.hail.annotations.aggregators.RegionValueAggregator
import is.hail.annotations._
import is.hail.expr.{SymbolTable, TypedAggregator}
import is.hail.expr.types._
import is.hail.methods._
import is.hail.utils._
import org.apache.spark.sql.Row

object Interpret {
  type Agg = (IndexedSeq[Row], TStruct)

  def apply[T](ir: IR): T = apply(ir, Env.empty[(Any, Type)], FastIndexedSeq(), None).asInstanceOf[T]

  def apply[T](ir: IR, optimize: Boolean): T = apply(ir, Env.empty[(Any, Type)], FastIndexedSeq(), None, optimize).asInstanceOf[T]

  def apply[T](ir0: IR,
    env: Env[(Any, Type)],
    args: IndexedSeq[(Any, Type)],
    agg: Option[Agg],
    optimize: Boolean = true): T = {
    val (typeEnv, valueEnv) = env.m.foldLeft((Env.empty[Type], Env.empty[Any])) {
      case ((e1, e2), (k, (value, t))) => (e1.bind(k, t), e2.bind(k, value))
    }

    var ir = ir0.unwrap
    if (optimize) {
      log.info("interpret: PRE-OPT\n" + Pretty(ir))
      ir = Optimize(ir)
      TypeCheck(ir, typeEnv, agg.map { agg =>
        agg._2.fields.foldLeft(Env.empty[Type]) { case (env, f) =>
          env.bind(f.name, f.typ)
        }
      })
    }

    apply(ir, valueEnv, args, agg, None).asInstanceOf[T]
  }

  private def apply(ir: IR, env: Env[Any], args: IndexedSeq[(Any, Type)], agg: Option[Agg], aggregator: Option[TypedAggregator[Any]]): Any = {
    def interpret(ir: IR, env: Env[Any] = env, args: IndexedSeq[(Any, Type)] = args, agg: Option[Agg] = agg, aggregator: Option[TypedAggregator[Any]] = aggregator): Any =
      apply(ir, env, args, agg, aggregator)
    ir match {
      case I32(x) => x
      case I64(x) => x
      case F32(x) => x
      case F64(x) => x
      case Str(x) => x
      case True() => true
      case False() => false
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
            case EQ(_, _) | EQWithNA(_, _) => lValue == rValue
            case NEQ(_, _) | NEQWithNA(_, _) => lValue != rValue
            case LT(t, _) => t.ordering.lt(lValue, rValue)
            case GT(t, _) => t.ordering.gt(lValue, rValue)
            case LTEQ(t, _) => t.ordering.lteq(lValue, rValue)
            case GTEQ(t, _) => t.ordering.gteq(lValue, rValue)
          }

      case MakeArray(elements, _) => elements.map(interpret(_, env, args, agg)).toIndexedSeq
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
          aValue.asInstanceOf[IndexedSeq[Row]].filter(_ != null).map{ case Row(k, v) => (k, v) }.toMap

      case ToArray(c) =>
        val ordering = coerce[TContainer](c.typ).elementType.ordering.toOrdering
        val cValue = interpret(c, env, args, agg)
        if (cValue == null)
          null
        else
          cValue match {
            case s: Set[_] =>
              s.toIndexedSeq.sorted(ordering)
            case d: Map[_, _] => d.iterator.map { case (k, v) => Row(k, v) }.toFastIndexedSeq.sorted(ordering)
            case a => a
          }

      case LowerBoundOnOrderedCollection(orderedCollection, elem, onKey) =>
        val cValue = interpret(orderedCollection, env, args, agg)
        val eValue = interpret(elem, env, args, agg)
        if (cValue == null)
          null
        else {
          val nSmaller = cValue match {
            case s: Set[_] =>
              assert(!onKey)
              s.count(elem.typ.ordering.lteq(_, eValue))
            case d: Map[_, _] =>
              assert(onKey)
              d.count { case (k, _) => elem.typ.ordering.lteq(k, eValue) }
            case a: IndexedSeq[_] =>
              assert(!onKey)
              a.count(elem.typ.ordering.lteq(_, eValue))
          }
          Integer.max(0, nSmaller - 1)
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
      case x@SeqOp(i, args, aggSig) =>
        assert(i == I32(0))
        aggSig.op match {
          case Inbreeding() =>
            val IndexedSeq(a, af) = args
            aggregator.get.asInstanceOf[InbreedingAggregator].seqOp(interpret(a), interpret(af))
          case TakeBy() =>
            val IndexedSeq(a, ordering) = args
            aggregator.get.asInstanceOf[TakeByAggregator[_]].seqOp(interpret(a), interpret(ordering))
          case Count() =>
            assert(args.isEmpty)
            aggregator.get.asInstanceOf[CountAggregator].seqOp(0) // 0 is a dummy value
          case _ =>
            val IndexedSeq(a) = args
            aggregator.get.seqOp(interpret(a))
        }
      case x@ApplyAggOp(a, constructorArgs, initOpArgs, aggSig) =>
        val seqOpArgTypes = aggSig.seqOpArgs
        assert(AggOp.getType(aggSig) == x.typ)
        val aggregator = aggSig.op match {
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
            val seqOps = Extract(a, _.isInstanceOf[SeqOp]).map(_.asInstanceOf[SeqOp])
            assert(seqOps.length == 1)
            val IndexedSeq(_, ordering: IR) = seqOps.head.args
            val ord = ordering.typ.ordering.toOrdering
            new TakeByAggregator(aggType, null, nValue)(ord)
          case Statistics() => new StatAggregator()
          case InfoScore() => new InfoScoreAggregator()
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
        }
        val Some((aggElements, aggElementType)) = agg
        aggElements.foreach { element =>
          val env = (element.toSeq, aggElementType.fieldNames).zipped
            .foldLeft(Env.empty[Any]) { case (env, (v, n)) =>
              env.bind(n, v)
            }
          interpret(a, env, FastIndexedSeq(), None, Some(aggregator))
        }
        aggregator.result
      case x@ApplyScanOp(a, constructorArgs, initOpArgs, aggSig) =>
        throw new UnsupportedOperationException("interpreter doesn't support scans right now.")
      case MakeStruct(fields) =>
        Row.fromSeq(fields.map { case (name, fieldIR) => interpret(fieldIR, env, args, agg) })
      case SelectFields(old, fields) =>
        val oldt = coerce[TStruct](old.typ)
        val oldRow = interpret(old, env, args, agg).asInstanceOf[Row]
        Row.fromSeq(fields.map(id => oldRow.get(oldt.fieldIdx(id))))
      case InsertFields(old, fields) =>
        var struct = interpret(old, env, args, agg)
        var t = old.typ
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
          if (vstart < 0 || vstart > vend || vend > vs.length)
            fatal(s"""string slice out of bounds or invalid: "$vs"[$vstart:$vend]""")
          else
            vs.substring(vstart, vend)
        }
      case StringLength(s) =>
        val vs = interpret(s).asInstanceOf[String]
        if (vs == null) null else vs.getBytes().length
      case In(i, _) =>
        val (a, _) = args(i)
        a
      case Die(message, typ) => fatal(message)
      case ir@ApplyIR(function, functionArgs, conversion) =>
        interpret(ir.explicitNode, env, args, agg)
      case ir@Apply(function, functionArgs) =>

        val argTuple = TTuple(functionArgs.map(_.typ): _*)
        val (_, makeFunction) = Compile[Long, Long]("in", argTuple, MakeTuple(List(Apply(function,
          functionArgs.zipWithIndex.map { case (x, i) =>
            GetTupleElement(Ref("in", argTuple), i)
          }))))

        val f = makeFunction()
        Region.scoped { region =>
          val rvb = new RegionValueBuilder()
          rvb.set(region)
          rvb.start(argTuple)
          rvb.startTuple()
          functionArgs.zip(argTuple.types).foreach { case (arg, t) =>
            val argValue = interpret(arg, env, args, agg)
            rvb.addAnnotation(t, argValue)
          }
          rvb.endTuple()
          val offset = rvb.end()

          val resultOffset = f(region, offset, false)
          SafeRow(TTuple(ir.implementation.returnType.subst()), region, resultOffset)
            .get(0)
        }
      case ir@ApplySpecial(function, functionArgs) =>
        val argTuple = TTuple(functionArgs.map(_.typ): _*)
        val (_, makeFunction) = Compile[Long, Long]("in", argTuple, MakeTuple(FastSeq(ApplySpecial(function,
          functionArgs.zipWithIndex.map { case (x, i) =>
            GetTupleElement(Ref("in", argTuple), i)
          }))))

        val f = makeFunction()
        Region.scoped { region =>
          val rvb = new RegionValueBuilder()
          rvb.set(region)
          rvb.start(argTuple)
          rvb.startTuple()
          functionArgs.zip(argTuple.types).foreach { case (arg, t) =>
            val argValue = interpret(arg, env, args, agg)
            rvb.addAnnotation(t, argValue)
          }
          rvb.endTuple()
          val offset = rvb.end()

          val resultOffset = f(region, offset, false)
          SafeRow(TTuple(ir.implementation.returnType.subst()), region, resultOffset)
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
      case MatrixWrite(child, f) =>
        val mv = child.execute(HailContext.get)
        f(mv)
      case TableWrite(child, path, overwrite, codecSpecJSONStr) =>
        val hc = HailContext.get
        val tableValue = child.execute(hc)
        tableValue.write(path, overwrite, codecSpecJSONStr)
      case TableExport(child, path, typesFile, header, exportType) =>
        val hc = HailContext.get
        val tableValue = child.execute(hc)
        tableValue.export(path, typesFile, header, exportType)
      case TableAggregate(child, query) =>
        val localGlobalSignature = child.typ.globalType
        val (rvAggs, initOps, seqOps, aggResultType, postAggIR) = CompileWithAggregators[Long, Long, Long](
          "global", child.typ.globalType,
          "global", child.typ.globalType,
          "row", child.typ.rowType,
          MakeTuple(Array(query)), "AGGR",
          (nAggs: Int, initOpIR: IR) => initOpIR,
          (nAggs: Int, seqOpIR: IR) => seqOpIR)

        val (t, f) = Compile[Long, Long, Long](
          "AGGR", aggResultType,
          "global", child.typ.globalType,
          postAggIR)

        val value = child.execute(HailContext.get)
        val globalsBc = value.globals.broadcast

        val aggResults = if (rvAggs.nonEmpty) {
          Region.scoped { region =>
            val rvb: RegionValueBuilder = new RegionValueBuilder()
            rvb.set(region)

            rvb.start(localGlobalSignature)
            rvb.addAnnotation(localGlobalSignature, globalsBc.value)
            val globals = rvb.end()

            initOps()(region, rvAggs, globals, false)
          }

          value.rvd.aggregate[Array[RegionValueAggregator]](rvAggs)({ case (rvaggs, rv) =>
            // add globals to region value
            val rowOffset = rv.offset
            val rvb = new RegionValueBuilder()
            rvb.set(rv.region)
            rvb.start(localGlobalSignature)
            rvb.addAnnotation(localGlobalSignature, globalsBc.value)
            val globalsOffset = rvb.end()

            seqOps()(rv.region, rvAggs, globalsOffset, false, rowOffset, false)
            rvAggs
          }, { (rvAggs1, rvAggs2) =>
            rvAggs1.zip(rvAggs2).foreach { case (rvAgg1, rvAgg2) => rvAgg1.combOp(rvAgg2) }
            rvAggs1
          })
        } else
          Array.empty[RegionValueAggregator]

        Region.scoped { region =>
          val rvb: RegionValueBuilder = new RegionValueBuilder()
          rvb.set(region)

          rvb.start(aggResultType)
          rvb.startStruct()
          aggResults.foreach(_.result(rvb))
          rvb.endStruct()
          val aggResultsOffset = rvb.end()

          rvb.start(localGlobalSignature)
          rvb.addAnnotation(localGlobalSignature, globalsBc.value)
          val globalsOffset = rvb.end()

          val resultOffset = f()(region, aggResultsOffset, false, globalsOffset, false)

          SafeRow(coerce[TTuple](t), region, resultOffset)
            .get(0)
        }
    }
  }
}

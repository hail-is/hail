package is.hail.expr.ir

import is.hail.HailContext
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
    if (optimize)
      ir = Optimize(ir)
    TypeCheck(ir, typeEnv, agg.map { agg =>
      agg._2.fields.foldLeft(Env.empty[Type]) { case (env, f) =>
          env.bind(f.name, f.typ)
      }
    })

    log.info("interpret:\n" + Pretty(ir))

    apply(ir, valueEnv, args, agg, None).asInstanceOf[T]
  }

  private def apply(ir: IR, env: Env[Any], args: IndexedSeq[Any], agg: Option[Agg], aggregator: Option[TypedAggregator[Any]]): Any = {
    def interpret(ir: IR, env: Env[Any] = env, args: IndexedSeq[Any] = args, agg: Option[Agg] = agg, aggregator: Option[TypedAggregator[Any]] = aggregator): Any =
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
                case GT() => ll > rr
                case GTEQ() => ll >= rr
                case LTEQ() => ll <= rr
                case LT() => ll < rr
                case EQ() => ll == rr
                case NEQ() => ll != rr
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
                case GT() => ll > rr
                case GTEQ() => ll >= rr
                case LTEQ() => ll <= rr
                case LT() => ll < rr
                case EQ() => ll == rr
                case NEQ() => ll != rr
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
                case GT() => ll > rr
                case GTEQ() => ll >= rr
                case LTEQ() => ll <= rr
                case LT() => ll < rr
                case EQ() => ll == rr
                case NEQ() => ll != rr
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
                case GT() => ll > rr
                case GTEQ() => ll >= rr
                case LTEQ() => ll <= rr
                case LT() => ll < rr
                case EQ() => ll == rr
                case NEQ() => ll != rr
              }
            case (_, _) =>
              op match {
                case EQ() => lValue == rValue
                case NEQ() => lValue != rValue
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
      case MakeArray(elements, _) => elements.map(interpret(_, env, args, agg)).toIndexedSeq
      case ArrayRef(a, i) =>
        val aValue = interpret(a, env, args, agg)
        val iValue = interpret(i, env, args, agg)
        if (aValue == null || iValue == null)
          null
        else
          aValue.asInstanceOf[IndexedSeq[Any]].apply(iValue.asInstanceOf[Int])
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
        if (startValue == null || stopValue == null || stepValue == null)
          null
        else
          startValue.asInstanceOf[Int] until stopValue.asInstanceOf[Int] by stepValue.asInstanceOf[Int]
      case ArraySort(a) =>
        val aValue = interpret(a, env, args, agg)
        if (aValue == null)
          null
        else
          aValue.asInstanceOf[IndexedSeq[Any]].sorted(a.typ.ordering.toOrdering)
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
        val cValue = interpret(c, env, args, agg)
        if (cValue == null)
          null
        else
          cValue match {
            case s: Set[Any] => s.toIndexedSeq
            case d: Map[Any, Any] => d.toIndexedSeq
            case a => a
          }

      case SetContains(s, elem) =>
        val sValue = interpret(s, env, args, agg)
        val eValue = interpret(elem, env, args, agg)
        if (sValue == null)
          null
        else
          sValue.asInstanceOf[Set[Any]].contains(eValue)

      case DictContains(d, key) =>
        val dValue = interpret(d, env, args, agg)
        val kValue = interpret(key, env, args, agg)
        if (dValue == null)
          null
        else
          dValue.asInstanceOf[Map[Any, Any]].contains(kValue)

      case DictGet(d, key) =>
        val dValue = interpret(d, env, args, agg)
        val kValue = interpret(key, env, args, agg)
        if (dValue == null)
          null
        else
          dValue.asInstanceOf[Map[Any, Any]].getOrElse(kValue, null)

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
      case x@SeqOp(a, i, aggSig) =>
        assert(i == I32(0))
        aggregator.get.seqOp(interpret(a))
      case x@ApplyAggOp(a, constructorArgs, initOpArgs, aggSig) =>
        val aggType = aggSig.inputType
        assert(AggOp.getType(aggSig) == x.typ)
        val aggregator = aggSig.op match {
          case CallStats() =>
            assert(aggType == TCall())
            val nAlleles = interpret(initOpArgs.get(0))
            new CallStatsAggregator(_ => nAlleles)
          case Collect() => new CollectAggregator(aggType)
          case Fraction() =>
            assert(aggType == TBoolean())
            new FractionAggregator(a => a)
          case Sum() =>
            aggType match {
              case TInt32(_) => new SumAggregator[Int]()
              case TInt64(_) => new SumAggregator[Long]()
              case TFloat32(_) => new SumAggregator[Float]()
              case TFloat64(_) => new SumAggregator[Double]()
            }
          case Max() =>
            aggType match {
              case TInt32(_) => new MaxAggregator[Int, java.lang.Integer]()
              case TInt64(_) => new MaxAggregator[Long, java.lang.Long]()
              case TFloat32(_) => new MaxAggregator[Float, java.lang.Float]()
              case TFloat64(_) => new MaxAggregator[Double, java.lang.Double]()
            }
          case Take() =>
            val Seq(n) = constructorArgs
            val nValue = interpret(n, Env.empty[Any], null, null).asInstanceOf[Int]
            new TakeAggregator(aggType, nValue)
          case Statistics() => new StatAggregator()
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
      case MakeStruct(fields) =>
        Row.fromSeq(fields.map { case (name, fieldIR) => interpret(fieldIR, env, args, agg) })
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
      case In(i, _) => args(i)
      case Die(message) => fatal(message)
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
          SafeRow(TTuple(ir.implementation.returnType), region, resultOffset)
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
          SafeRow(TTuple(ir.implementation.returnType), region, resultOffset)
            .get(0)
        }
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
        val (rvAggs, initOps, seqOps, aggResultType, f, t) = CompileWithAggregators[Long, Long, Long, Long](
          "global", child.typ.globalType,
          "global", child.typ.globalType,
          "row", child.typ.rowType,
          MakeTuple(Array(query)),
          (nAggs: Int, initOpIR: IR) => initOpIR,
          (nAggs: Int, seqOpIR: IR) => seqOpIR)

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

package is.hail.expr.ir

import is.hail.HailContext
import is.hail.annotations.aggregators.RegionValueAggregator
import is.hail.annotations.{Region, RegionValueBuilder, UnsafeRow}
import is.hail.expr.types._
import is.hail.methods._
import is.hail.utils._
import org.apache.spark.sql.Row

object Interpret {
  type AggElement = (Any, Env[Any])
  type Agg = (TAggregable, IndexedSeq[AggElement])

  def apply[T](ir: IR): T = apply(ir, Env.empty[(Any, Type)], IndexedSeq(), None).asInstanceOf[T]

  def apply[T](ir: IR,
    env: Env[(Any, Type)],
    args: IndexedSeq[(Any, Type)],
    agg: Option[Agg]): T = {
    val (typeEnv, valueEnv) = env.m.foldLeft((Env.empty[Type], Env.empty[Any])) {
      case ((e1, e2), (k, (value, t))) => (e1.bind(k, t), e2.bind(k, value))
    }

    Infer(ir, agg.map(_._1), typeEnv)
    interpret(ir, valueEnv, args, agg.orNull).asInstanceOf[T]
  }

  private def interpret(ir: IR, env: Env[Any], args: IndexedSeq[Any], agg: Agg): Any = {
    ir match {
      case I32(x) => x
      case I64(x) => x
      case F32(x) => x
      case F64(x) => x
      case True() => true
      case False() => false
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
      case If(cond, cnsq, altr, _) =>
        val condValue = interpret(cond, env, args, agg)
        if (condValue == null)
          null
        else if (condValue.asInstanceOf[Boolean])
          interpret(cnsq, env, args, agg)
        else
          interpret(altr, env, args, agg)
      case Let(name, value, body, _) =>
        val valueValue = interpret(value, env, args, agg)
        interpret(body, env.bind(name, valueValue), args, agg)
      case Ref(name, _) => env.lookup(name)
      case ApplyBinaryPrimOp(op, l, r, _) =>
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
      case ApplyUnaryPrimOp(op, x, _) =>
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
      case ArrayRef(a, i, _) =>
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
      case ArrayMap(a, name, body, _) =>
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
      case ArrayFold(a, zero, accumName, valueName, body, _) =>
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
      case x@ApplyAggOp(a, op, aggArgs, _) =>
        val aValue = interpretAgg(a, agg._2)
        val aggType = a.typ.asInstanceOf[TAggregable].elementType
        assert(aValue != null)
        assert(AggOp.getType(op, x.inputType, aggArgs.map(_.typ)) == x.typ)
        val aggregator = op match {
          case Collect() => new CollectAggregator(aggType)
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
            val Seq(n) = aggArgs
            val nValue = interpret(n, Env.empty[Any], null, null).asInstanceOf[Int]
            new TakeAggregator(aggType, nValue)
          case Statistics() => new StatAggregator()
          case Histogram() =>
            val Seq(start, end, bins) = aggArgs
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
        aValue.foreach { case (element, _) =>
            aggregator.seqOp(element)
        }
        aggregator.result
      case MakeStruct(fields, _) =>
        Row.fromSeq(fields.map { case (name, fieldIR) => interpret(fieldIR, env, args, agg) })
      case InsertFields(old, fields, _) =>
        var struct = interpret(old, env, args, agg)
        var t = old.typ
        fields.foreach { case (name, body) =>
          val (newT, ins) = t.insert(body.typ, name)
          t = newT.asInstanceOf[TStruct]
          struct = ins(struct, interpret(body, env, args, agg))
        }
        struct
      case GetField(o, name, _) =>
        val oValue = interpret(o, env, args, agg)
        if (oValue == null)
          null
        else {
          val oType = o.typ.asInstanceOf[TStruct]
          val fieldIndex = oType.fieldIdx(name)
          oValue.asInstanceOf[Row].get(fieldIndex)
        }
      case MakeTuple(types, _) =>
        Row.fromSeq(types.map(x => interpret(x, env, args, agg)))
      case GetTupleElement(o, idx, _) =>
        val oValue = interpret(o, env, args, agg)
        if (oValue == null)
          null
        else
          oValue.asInstanceOf[Row].get(idx)
      case In(i, _) => args(i)
      case Die(message) => fatal(message)
      case Apply(function, functionArgs, implementation) =>
        val argTuple = TTuple(functionArgs.map(_.typ): _*)
        val (_, makeFunction) = Compile[Long, Long]("in", argTuple, MakeTuple(List(Apply(function,
          (0 until argTuple.size)
            .map(i => GetTupleElement(Ref("in"), i))))))

        val f = makeFunction()
        val region = Region()
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
        val ur = new UnsafeRow(TTuple(implementation.returnType), region, resultOffset)
        ur.get(0)
      case ApplySpecial(function, functionArgs, implementation) =>
        val argTuple = TTuple(functionArgs.map(_.typ): _*)
        val (_, makeFunction) = Compile[Long, Long]("in", argTuple, MakeTuple(List(ApplySpecial(function,
          (0 until argTuple.size)
            .map(i => GetTupleElement(Ref("in"), i))))))

        val f = makeFunction()
        val region = Region()
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
        val ur = new UnsafeRow(TTuple(implementation.returnType), region, resultOffset)
        ur.get(0)
      case TableCount(child) =>
        child.partitionCounts
          .map(_.sum)
          .getOrElse(child.execute(HailContext.get).rvd.count())
      case TableWrite(child, path, overwrite, codecSpecJSONStr) =>
        val hc = HailContext.get
        val tableValue = child.execute(hc)
        tableValue.write(path, overwrite, codecSpecJSONStr)
      case TableAggregate(child, query, _) =>
        val localGlobalSignature = child.typ.globalType
        val tAgg = child.typ.aggEnv.lookup("AGG").asInstanceOf[TAggregable]

        val (rvAggs, seqOps, aggResultType, f, t) = CompileWithAggregators[Long, Long, Long, Long, Long](
          "AGG", tAgg,
          "global", child.typ.globalType,
          MakeTuple(Array(query)))

        val value = child.execute(HailContext.get)
        val globalsBc = value.globals.broadcast
        val aggResults = if (seqOps.nonEmpty) {
          value.rvd.aggregate[Array[RegionValueAggregator]](rvAggs)({ case (rvaggs, rv) =>
            // add globals to region value
            val rowOffset = rv.offset
            val rvb = new RegionValueBuilder()
            rvb.set(rv.region)
            rvb.start(localGlobalSignature)
            rvb.addAnnotation(localGlobalSignature, globalsBc.value)
            val globalsOffset = rvb.end()

            rvaggs.zip(seqOps).foreach { case (rvagg, seqOp) =>
              seqOp()(rv.region, rvagg, rowOffset, false, globalsOffset, false, rowOffset, false)
            }
            rvaggs
          }, { (rvAggs1, rvAggs2) =>
            rvAggs1.zip(rvAggs2).foreach { case (rvAgg1, rvAgg2) => rvAgg1.combOp(rvAgg2) }
            rvAggs1
          })
        } else
          Array.empty[RegionValueAggregator]

        val region: Region = Region()
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
        val resultType = coerce[TTuple](t)
        UnsafeRow.readBaseStruct(resultType, region, resultOffset).get(0)
    }
  }

  private def interpretAgg(ir: IR, agg: IndexedSeq[AggElement]): IndexedSeq[AggElement] = {
    (ir: @unchecked) match {
      case AggIn(_) =>
        assert(agg != null)
        agg
      case AggMap(a, name, body, t) =>
        interpretAgg(a, agg)
          .map { case (element, env) =>
            interpret(body, env.bind(name, element), null, null) -> env
          }
      case AggFilter(a, name, body, _) =>
        interpretAgg(a, agg)
          .filter { case (element, env) =>
            // casting to boolean treats null as false
            interpret(body, env.bind(name, element), null, null).asInstanceOf[Boolean]
          }
      case AggFlatMap(a, name, body, _) =>
        interpretAgg(a, agg)
          .flatMap { case (element, env) =>
            interpret(body, env.bind(name, element), null, null)
              .asInstanceOf[Iterable[Any]]
              .map(_ -> env)
          }
    }
  }
}

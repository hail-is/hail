package is.hail.expr.ir

import is.hail.annotations._
import is.hail.asm4s.AsmFunction3
import is.hail.expr.TypedAggregator
import is.hail.expr.ir.lowering.LoweringPipeline
import is.hail.expr.types.physical.{PTuple, PType}
import is.hail.expr.types.virtual._
import is.hail.io.BufferSpec
import is.hail.linalg.BlockMatrix
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

  def apply[T](ctx: ExecuteContext, ir: IR): T = apply(ctx, ir, Env.empty[(Any, Type)], FastIndexedSeq[(Any, Type)]()).asInstanceOf[T]

  def apply[T](ctx: ExecuteContext,
    ir0: IR,
    env: Env[(Any, Type)],
    args: IndexedSeq[(Any, Type)],
    optimize: Boolean = true
  ): T = {
    val rwIR = env.m.foldLeft[IR](ir0) { case (acc, (k, (value, t))) => Let(k, Literal.coerce(t, value), acc) }

    val lowered = LoweringPipeline.relationalLowerer.apply(ctx, rwIR, optimize).asInstanceOf[IR]

    val result = run(ctx, lowered, Env.empty[Any], args, Memo.empty).asInstanceOf[T]

    result
  }

  def alreadyLowered(ctx: ExecuteContext, ir: IR): Any = run(ctx, ir, Env.empty, FastIndexedSeq(), Memo.empty)

  private def run(ctx: ExecuteContext,
    ir: IR,
    env: Env[Any],
    args: IndexedSeq[(Any, Type)],
    functionMemo: Memo[(PType, AsmFunction3[Region, Long, Boolean, Long])]): Any = {

    def interpret(ir: IR, env: Env[Any] = env, args: IndexedSeq[(Any, Type)] = args): Any =
      run(ctx, ir, env, args, functionMemo)

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
        val vValue = interpret(v, env, args)
        if (vValue == null)
          null
        else
          (v.typ, t) match {
            case (TInt32, TInt32) => vValue
            case (TInt32, TInt64) => vValue.asInstanceOf[Int].toLong
            case (TInt32, TFloat32) => vValue.asInstanceOf[Int].toFloat
            case (TInt32, TFloat64) => vValue.asInstanceOf[Int].toDouble
            case (TInt64, TInt64) => vValue
            case (TInt64, TInt32) => vValue.asInstanceOf[Long].toInt
            case (TInt64, TFloat32) => vValue.asInstanceOf[Long].toFloat
            case (TInt64, TFloat64) => vValue.asInstanceOf[Long].toDouble
            case (TFloat32, TFloat32) => vValue
            case (TFloat32, TInt32) => vValue.asInstanceOf[Float].toInt
            case (TFloat32, TInt64) => vValue.asInstanceOf[Float].toLong
            case (TFloat32, TFloat64) => vValue.asInstanceOf[Float].toDouble
            case (TFloat64, TFloat64) => vValue
            case (TFloat64, TInt32) => vValue.asInstanceOf[Double].toInt
            case (TFloat64, TInt64) => vValue.asInstanceOf[Double].toLong
            case (TFloat64, TFloat32) => vValue.asInstanceOf[Double].toFloat
            case (TInt32, TCall) => vValue
          }
      case CastRename(v, _) => interpret(v)
      case NA(_) => null
      case IsNA(value) => interpret(value, env, args) == null
      case Coalesce(values) =>
        values.iterator
          .flatMap(x => Option(interpret(x, env, args)))
          .headOption
          .orNull
      case If(cond, cnsq, altr) =>
        assert(cnsq.typ == altr.typ)
        val condValue = interpret(cond, env, args)
        if (condValue == null)
          null
        else if (condValue.asInstanceOf[Boolean])
          interpret(cnsq, env, args)
        else
          interpret(altr, env, args)
      case Let(name, value, body) =>
        val valueValue = interpret(value, env, args)
        interpret(body, env.bind(name, valueValue), args)
      case Ref(name, _) => env.lookup(name)
      case ApplyBinaryPrimOp(op, l, r) =>
        val lValue = interpret(l, env, args)
        val rValue = interpret(r, env, args)
        if (lValue == null || rValue == null)
          null
        else
          (l.typ, r.typ) match {
            case (TInt32, TInt32) =>
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
            case (TInt64, TInt32) =>
              val ll = lValue.asInstanceOf[Long]
              val rr = rValue.asInstanceOf[Int]
              (op: @unchecked) match {
                case LeftShift() => ll << rr
                case RightShift() => ll >> rr
                case LogicalRightShift() => ll >>> rr
              }
            case (TInt64, TInt64) =>
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
            case (TFloat32, TFloat32) =>
              val ll = lValue.asInstanceOf[Float]
              val rr = rValue.asInstanceOf[Float]
              (op: @unchecked) match {
                case Add() => ll + rr
                case Subtract() => ll - rr
                case Multiply() => ll * rr
                case FloatingPointDivide() => ll / rr
                case RoundToNegInfDivide() => math.floor(ll / rr).toFloat
              }
            case (TFloat64, TFloat64) =>
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
        val xValue = interpret(x, env, args)
        if (xValue == null)
          null
        else op match {
          case Bang() =>
            assert(x.typ == TBoolean)
            !xValue.asInstanceOf[Boolean]
          case Negate() =>
            assert(x.typ.isInstanceOf[TNumeric])
            x.typ match {
              case TInt32 => -xValue.asInstanceOf[Int]
              case TInt64 => -xValue.asInstanceOf[Long]
              case TFloat32 => -xValue.asInstanceOf[Float]
              case TFloat64 => -xValue.asInstanceOf[Double]
            }
          case BitNot() =>
            x.typ match {
              case TInt32 => ~xValue.asInstanceOf[Int]
              case TInt64 => ~xValue.asInstanceOf[Long]
            }
        }
      case ApplyComparisonOp(op, l, r) =>
        val lValue = interpret(l, env, args)
        val rValue = interpret(r, env, args)
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

      case MakeArray(elements, _) => elements.map(interpret(_, env, args)).toFastIndexedSeq
      case MakeStream(elements, _) => elements.map(interpret(_, env, args)).toFastIndexedSeq
      case x@ArrayRef(a, i, s) =>
        val aValue = interpret(a, env, args)
        val iValue = interpret(i, env, args)
        if (aValue == null || iValue == null)
          null
        else {
          val a = aValue.asInstanceOf[IndexedSeq[Any]]
          val i = iValue.asInstanceOf[Int]

          if (i < 0 || i >= a.length) {
            val msg = interpret(s, env, args)
            val prettied = Pretty(x)
            val irString =
              if (prettied.size > 100) prettied.take(100) + " ..."
              else prettied
            val toAdd = if (msg == "") "" else s"\n----------\nPython traceback:\n${ msg }"
            fatal(s"array index out of bounds: index=$i, length=${ a.length }" +
              s"\n----------\nIR:\n${ irString }$s" + toAdd)
          } else
            a.apply(i)
        }
      case ArrayLen(a) =>
        val aValue = interpret(a, env, args)
        if (aValue == null)
          null
        else
          aValue.asInstanceOf[IndexedSeq[Any]].length
      case StreamRange(start, stop, step) =>
        val startValue = interpret(start, env, args)
        val stopValue = interpret(stop, env, args)
        val stepValue = interpret(step, env, args)
        if (stepValue == 0)
          fatal("Array range cannot have step size 0.")
        if (startValue == null || stopValue == null || stepValue == null)
          null
        else
          startValue.asInstanceOf[Int] until stopValue.asInstanceOf[Int] by stepValue.asInstanceOf[Int]
      case ArraySort(a, l, r, compare) =>
        val aValue = interpret(a, env, args)
        if (aValue == null)
          null
        else {
          aValue.asInstanceOf[IndexedSeq[Any]].sortWith { (left, right) =>
            if (left != null && right != null) {
              val res = interpret(compare, env.bind(l, left).bind(r, right), args)
              if (res == null)
                fatal("Result of sorting function cannot be missing.")
              res.asInstanceOf[Boolean]
            } else {
              right == null
            }
          }
        }
      case ToSet(a) =>
        val aValue = interpret(a, env, args)
        if (aValue == null)
          null
        else
          aValue.asInstanceOf[IndexedSeq[Any]].toSet
      case ToDict(a) =>
        val aValue = interpret(a, env, args)
        if (aValue == null)
          null
        else
          aValue.asInstanceOf[IndexedSeq[Row]].filter(_ != null).map { case Row(k, v) => (k, v) }.toMap
      case _: CastToArray | _: ToArray | _: ToStream =>
        val c = ir.children(0).asInstanceOf[IR]
        val cValue = interpret(c, env, args)
        if (cValue == null)
          null
        else {
          val ordering = coerce[TIterable](c.typ).elementType.ordering.toOrdering
          cValue match {
            case s: Set[_] =>
              s.asInstanceOf[Set[Any]].toFastIndexedSeq.sorted(ordering)
            case d: Map[_, _] => d.iterator.map { case (k, v) => Row(k, v) }.toFastIndexedSeq.sorted(ordering)
            case a => a
          }
        }

      case LowerBoundOnOrderedCollection(orderedCollection, elem, onKey) =>
        val cValue = interpret(orderedCollection, env, args)
        val eValue = interpret(elem, env, args)
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
        interpret(collection, env, args).asInstanceOf[IndexedSeq[Row]]
          .groupBy { case Row(k, _) => k }
          .mapValues { elt: IndexedSeq[Row] => elt.map { case Row(_, v) => v } }

      case StreamMap(a, name, body) =>
        val aValue = interpret(a, env, args)
        if (aValue == null)
          null
        else {
          aValue.asInstanceOf[IndexedSeq[Any]].map { element =>
            interpret(body, env.bind(name, element), args)
          }
        }
      case StreamZip(as, names, body, behavior) =>
        val aValues = as.map(interpret(_, env, args).asInstanceOf[IndexedSeq[_]])
        if (aValues.contains(null))
          null
        else {
          val len = behavior match {
            case ArrayZipBehavior.AssertSameLength | ArrayZipBehavior.AssumeSameLength =>
              val lengths = aValues.map(_.length).toSet
              if (lengths.size != 1)
                fatal(s"zip: length mismatch: ${ lengths.mkString(", ") }")
              lengths.head
            case ArrayZipBehavior.TakeMinLength =>
              aValues.map(_.length).min
            case ArrayZipBehavior.ExtendNA =>
              aValues.map(_.length).max
          }
          (0 until len).map { i =>
            val e = env.bindIterable(names.zip(aValues.map(a => if (i >= a.length) null else a.apply(i))))
            interpret(body, e, args)
          }
        }
      case StreamFilter(a, name, cond) =>
        val aValue = interpret(a, env, args)
        if (aValue == null)
          null
        else {
          aValue.asInstanceOf[IndexedSeq[Any]].filter { element =>
            // casting to boolean treats null as false
            interpret(cond, env.bind(name, element), args).asInstanceOf[Boolean]
          }
        }
      case StreamFlatMap(a, name, body) =>
        val aValue = interpret(a, env, args)
        if (aValue == null)
          null
        else {
          aValue.asInstanceOf[IndexedSeq[Any]].flatMap { element =>
            val r = interpret(body, env.bind(name, element), args).asInstanceOf[IndexedSeq[Any]]
            if (r != null)
              r
            else
              None
          }
        }
      case StreamFold(a, zero, accumName, valueName, body) =>
        val aValue = interpret(a, env, args)
        if (aValue == null)
          null
        else {
          var zeroValue = interpret(zero, env, args)
          aValue.asInstanceOf[IndexedSeq[Any]].foreach { element =>
            zeroValue = interpret(body, env.bind(accumName -> zeroValue, valueName -> element), args)
          }
          zeroValue
        }
      case StreamFold2(a, accum, valueName, seq, res) =>
        val aValue = interpret(a, env, args)
        if (aValue == null)
          null
        else {
          val accVals = accum.map { case (name, value) => (name, interpret(value, env, args)) }
          var e = env.bindIterable(accVals)
          aValue.asInstanceOf[IndexedSeq[Any]].foreach { elt =>
            e = e.bind(valueName, elt)
            accVals.indices.foreach { i =>
              e = e.bind(accum(i)._1, interpret(seq(i), e, args))
            }
          }
          interpret(res, e.delete(valueName), args)
        }
      case StreamScan(a, zero, accumName, valueName, body) =>
        val aValue = interpret(a, env, args)
        if (aValue == null)
          null
        else {
          val zeroValue = interpret(zero, env, args)
          aValue.asInstanceOf[IndexedSeq[Any]].scanLeft(zeroValue) { (accum, elt) =>
            interpret(body, env.bind(accumName -> accum, valueName -> elt), args)
          }
        }

      case StreamLeftJoinDistinct(left, right, l, r, compare, join) =>
        val lValue = interpret(left, env, args).asInstanceOf[IndexedSeq[Any]]
        val rValue = interpret(right, env, args).asInstanceOf[IndexedSeq[Any]].toIterator

        var relt: Any = if (rValue.hasNext) rValue.next() else null

        lValue.map { lelt =>
          while (rValue.hasNext && interpret(compare, env.bind(l -> lelt, r -> relt), args).asInstanceOf[Int] > 0) {
            relt = rValue.next()
          }
          if (interpret(compare, env.bind(l -> lelt, r -> relt), args).asInstanceOf[Int] == 0) {
            interpret(join, env.bind(l -> lelt, r -> relt), args)
          } else {
            interpret(join, env.bind(l -> lelt, r -> null), args)
          }
        }

      case StreamFor(a, valueName, body) =>
        val aValue = interpret(a, env, args)
        if (aValue != null) {
          aValue.asInstanceOf[IndexedSeq[Any]].foreach { element =>
            interpret(body, env.bind(valueName -> element), args)
          }
        }
        ()
      case Begin(xs) =>
        xs.foreach(x => interpret(x))
      case MakeStruct(fields) =>
        Row.fromSeq(fields.map { case (name, fieldIR) => interpret(fieldIR, env, args) })
      case SelectFields(old, fields) =>
        val oldt = coerce[TStruct](old.typ)
        val oldRow = interpret(old, env, args).asInstanceOf[Row]
        if (oldRow == null)
          null
        else
          Row.fromSeq(fields.map(id => oldRow.get(oldt.fieldIdx(id))))
      case x@InsertFields(old, fields, fieldOrder) =>
        var struct = interpret(old, env, args)
        if (struct != null)
          fieldOrder match {
            case Some(fds) =>
              val newValues = fields.toMap.mapValues(interpret(_, env, args))
              val oldIndices = old.typ.asInstanceOf[TStruct].fields.map(f => f.name -> f.index).toMap
              Row.fromSeq(fds.map(name => newValues.getOrElse(name, struct.asInstanceOf[Row].get(oldIndices(name)))))
            case None =>
              var t = old.typ
              fields.foreach { case (name, body) =>
                val (newT, ins) = t.insert(body.typ, name)
                t = newT.asInstanceOf[TStruct]
                struct = ins(struct, interpret(body, env, args))
              }
              struct
          }
        else
          null

      case GetField(o, name) =>
        val oValue = interpret(o, env, args)
        if (oValue == null)
          null
        else {
          val oType = o.typ.asInstanceOf[TStruct]
          val fieldIndex = oType.fieldIdx(name)
          oValue.asInstanceOf[Row].get(fieldIndex)
        }
      case MakeTuple(types) =>
        Row.fromSeq(types.map { case (_, x) => interpret(x, env, args) })
      case GetTupleElement(o, idx) =>
        val oValue = interpret(o, env, args)
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
        interpret(ir.explicitNode, env, args)
      case ApplySpecial("lor", Seq(left_, right_), _) =>
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
      case ApplySpecial("land", Seq(left_, right_), _) =>
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
            val argValue = interpret(arg, env, args)
            rvb.addAnnotation(t.virtualType, argValue)
          }
          rvb.endTuple()
          val offset = rvb.end()

          try {
            val resultOffset = f(region, offset, false)
            SafeRow(rt.asInstanceOf[PTuple], resultOffset).get(0)
          } catch {
            case e: Exception =>
              fatal(s"error while calling '${ ir.implementation.name }'", e)
          }
        }
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
      case UnpersistBlockMatrix(BlockMatrixRead(BlockMatrixPersistReader(id))) =>
        HailContext.backend.cache.unpersistBlockMatrix(id)
      case _: UnpersistBlockMatrix =>
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
            SafeRow(rt, f(0, region)(region, globalsOffset, false))
          }
        } else {
          val spec = BufferSpec.defaultUncompressed

          val physicalAggs = extracted.getPhysicalAggs(
            ctx,
            Env("global" -> value.globals.t),
            Env("global" -> value.globals.t, "row" -> value.rvd.rowPType)
          )

          val (_, initOp) = CompileWithAggregators2[Long, Unit](ctx,
            physicalAggs,
            "global", value.globals.t,
            extracted.init)

          val (_, partitionOpSeq) = CompileWithAggregators2[Long, Long, Unit](ctx,
            physicalAggs,
            "global", value.globals.t,
            "row", value.rvd.rowPType,
            extracted.seqPerElt)

          val read = extracted.deserialize(ctx, spec, physicalAggs)
          val write = extracted.serialize(ctx, spec, physicalAggs)
          val combOpF = extracted.combOpF(ctx, spec, physicalAggs)

          val (rTyp: PTuple, f) = CompileWithAggregators2[Long, Long](ctx,
            physicalAggs,
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
              val partRegion = ctx.partitionRegion
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
              SafeRow(rTyp, resF(r, globalsOffset, false))
            }
          }
        }

        wrapped.get(0)
      case LiftMeOut(child) =>
        val (rt, makeFunction) = Compile[Long](ctx, MakeTuple.ordered(FastSeq(child)), None, false)
        Region.scoped { r =>
          SafeRow.read(rt, makeFunction(0, r)(r)).asInstanceOf[Row](0)
        }

    }
  }
}

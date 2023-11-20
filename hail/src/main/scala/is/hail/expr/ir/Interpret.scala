package is.hail.expr.ir

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend.spark.SparkTaskContext
import is.hail.backend.{ExecuteContext, HailTaskContext}
import is.hail.expr.ir.lowering.LoweringPipeline
import is.hail.io.BufferSpec
import is.hail.linalg.BlockMatrix
import is.hail.rvd.RVDContext
import is.hail.types.physical.stypes.{PTypeReferenceSingleCodeType, SingleCodeType}
import is.hail.types.physical.{PTuple, PType}
import is.hail.types.tcoerce
import is.hail.types.virtual._
import is.hail.utils._
import org.apache.spark.sql.Row

import scala.collection.mutable

object Interpret {
  type Agg = (IndexedSeq[Row], TStruct)

  def apply(tir: TableIR, ctx: ExecuteContext): TableValue =
    apply(tir, ctx, optimize = true)

  def apply(tir: TableIR, ctx: ExecuteContext, optimize: Boolean): TableValue = {
    val lowered = LoweringPipeline.legacyRelationalLowerer(optimize)(ctx, tir).asInstanceOf[TableIR].noSharing(ctx)
    lowered.analyzeAndExecute(ctx).asTableValue(ctx)
  }

  def apply(mir: MatrixIR, ctx: ExecuteContext, optimize: Boolean): TableValue = {
    val lowered = LoweringPipeline.legacyRelationalLowerer(optimize)(ctx, mir).asInstanceOf[TableIR]
    lowered.analyzeAndExecute(ctx).asTableValue(ctx)
  }

  def apply(bmir: BlockMatrixIR, ctx: ExecuteContext, optimize: Boolean): BlockMatrix = {
    val lowered = LoweringPipeline.legacyRelationalLowerer(optimize)(ctx, bmir).asInstanceOf[BlockMatrixIR]
    lowered.execute(ctx)
  }

  def apply[T](ctx: ExecuteContext, ir: IR): T = apply(ctx, ir, Env.empty[(Any, Type)], FastSeq[(Any, Type)]()).asInstanceOf[T]

  def apply[T](ctx: ExecuteContext,
    ir0: IR,
    env: Env[(Any, Type)],
    args: IndexedSeq[(Any, Type)],
    optimize: Boolean = true
  ): T = {
    val rwIR = env.m.foldLeft[IR](ir0) { case (acc, (k, (value, t))) => Let(k, Literal.coerce(t, value), acc) }

    val lowered = LoweringPipeline.relationalLowerer(optimize).apply(ctx, rwIR).asInstanceOf[IR]

    val result = run(ctx, lowered, Env.empty[Any], args, Memo.empty).asInstanceOf[T]

    result
  }

  def alreadyLowered(ctx: ExecuteContext, ir: IR): Any = run(ctx, ir, Env.empty, FastSeq(), Memo.empty)

  private def run(ctx: ExecuteContext,
    ir: IR,
    env: Env[Any],
    args: IndexedSeq[(Any, Type)],
    functionMemo: Memo[(SingleCodeType, AsmFunction2RegionLongLong)]): Any = {

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
      case x@EncodedLiteral(codec, value) =>
        ctx.r.getPool().scopedRegion { r =>
          val (pt, addr) = codec.decodeArrays(ctx, x.typ, value.ba, ctx.r)
          SafeRow.read(pt, addr)
        }
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
      case Switch(x_, default, cases) =>
        interpret(x_, env, args) match {
          case x: Int =>
            interpret(if (x >= 0 && x < cases.length) cases(x) else default, env, args)
          case null =>
            null
        }
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
                case FloatingPointDivide() => ll.toDouble / rr.toDouble
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
                case FloatingPointDivide() => ll.toDouble / rr.toDouble
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
          case Bang =>
            assert(x.typ == TBoolean)
            !xValue.asInstanceOf[Boolean]
          case Negate =>
            assert(x.typ.isInstanceOf[TNumeric])
            x.typ match {
              case TInt32 => -xValue.asInstanceOf[Int]
              case TInt64 => -xValue.asInstanceOf[Long]
              case TFloat32 => -xValue.asInstanceOf[Float]
              case TFloat64 => -xValue.asInstanceOf[Double]
            }
          case BitNot =>
            assert(x.typ.isInstanceOf[TIntegral])
            x.typ match {
              case TInt32 => ~xValue.asInstanceOf[Int]
              case TInt64 => ~xValue.asInstanceOf[Long]
            }
          case BitCount =>
            assert(x.typ.isInstanceOf[TIntegral])
            x.typ match {
              case TInt32 => Integer.bitCount(xValue.asInstanceOf[Int])
              case TInt64 => java.lang.Long.bitCount(xValue.asInstanceOf[Long])
            }
        }
      case ApplyComparisonOp(op, l, r) =>
        val lValue = interpret(l, env, args)
        val rValue = interpret(r, env, args)
        if (op.strict && (lValue == null || rValue == null))
          null
        else
          op match {
            case EQ(t, _) => t.ordering(ctx.stateManager).equiv(lValue, rValue)
            case EQWithNA(t, _) => t.ordering(ctx.stateManager).equiv(lValue, rValue)
            case NEQ(t, _) => !t.ordering(ctx.stateManager).equiv(lValue, rValue)
            case NEQWithNA(t, _) => !t.ordering(ctx.stateManager).equiv(lValue, rValue)
            case LT(t, _) => t.ordering(ctx.stateManager).lt(lValue, rValue)
            case GT(t, _) => t.ordering(ctx.stateManager).gt(lValue, rValue)
            case LTEQ(t, _) => t.ordering(ctx.stateManager).lteq(lValue, rValue)
            case GTEQ(t, _) => t.ordering(ctx.stateManager).gteq(lValue, rValue)
            case Compare(t, _) => t.ordering(ctx.stateManager).compare(lValue, rValue)
          }

      case MakeArray(elements, _) => elements.map(interpret(_, env, args)).toFastSeq
      case MakeStream(elements, _, _) => elements.map(interpret(_, env, args)).toFastSeq
      case x@ArrayRef(a, i, errorId) =>
        val aValue = interpret(a, env, args)
        val iValue = interpret(i, env, args)
        if (aValue == null || iValue == null)
          null
        else {
          val a = aValue.asInstanceOf[IndexedSeq[Any]]
          val i = iValue.asInstanceOf[Int]

          if (i < 0 || i >= a.length) {
            fatal(s"array index out of bounds: index=$i, length=${ a.length }", errorId = errorId)
          } else
            a.apply(i)
        }
      case ArraySlice(a, start, stop, step, errorID) =>
        val aValue = interpret(a, env, args)
        val startValue = interpret(start, env, args)
        val stopValue = stop.map(ir => interpret(ir, env, args))
        val stepValue = interpret(step, env, args)
        if (startValue == null || stepValue == null || aValue == null  ||
          stopValue.getOrElse(aValue.asInstanceOf[IndexedSeq[Any]].size) == null)
          null
        else {
          val a = aValue.asInstanceOf[IndexedSeq[Any]]
          val requestedStart = startValue.asInstanceOf[Int]
          val requestedStep = stepValue.asInstanceOf[Int]
          if (requestedStep == 0)
            fatal("step cannot be 0 for array slice", errorID)
          val noneStop = if (requestedStep < 0) -a.size - 1
            else a.size
          val maxBound = if(requestedStep > 0) a.size
            else a.size - 1
          val minBound = if(requestedStep > 0) 0
            else - 1
          val requestedStop = stopValue.getOrElse(noneStop).asInstanceOf[Int]
          val realStart = if (requestedStart >= a.size) maxBound
            else if (requestedStart >= 0) requestedStart
            else if (requestedStart + a.size >= 0) requestedStart + a.size
            else minBound
          val realStop = if (requestedStop >= a.size) maxBound
            else if (requestedStop >= 0) requestedStop
            else if (requestedStop + a.size > 0) requestedStop + a.size
            else minBound
          (realStart until realStop by requestedStep).map(idx => a(idx))
        }
      case ArrayLen(a) =>
        val aValue = interpret(a, env, args)
        if (aValue == null)
          null
        else
          aValue.asInstanceOf[IndexedSeq[Any]].length
      case StreamLen(a) =>
        val aValue = interpret(a, env, args)
        if (aValue == null)
          null
        else
          aValue.asInstanceOf[IndexedSeq[Any]].length
      case StreamIota(start, step, requiresMemoryManagementPerElement) => throw new UnsupportedOperationException
      case StreamRange(start, stop, step, _, errorID) =>
        val startValue = interpret(start, env, args)
        val stopValue = interpret(stop, env, args)
        val stepValue = interpret(step, env, args)
        if (stepValue == 0)
          fatal("Array range cannot have step size 0.", errorID)
        if (startValue == null || stopValue == null || stepValue == null)
          null
        else
          startValue.asInstanceOf[Int] until stopValue.asInstanceOf[Int] by stepValue.asInstanceOf[Int]
      case ArraySort(a, l, r, lessThan) =>
        val aValue = interpret(a, env, args)
        if (aValue == null)
          null
        else {
          aValue.asInstanceOf[IndexedSeq[Any]].sortWith { (left, right) =>
            if (left != null && right != null) {
              val res = interpret(lessThan, env.bind(l, left).bind(r, right), args)
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
        val c = ir.children.head.asInstanceOf[IR]
        val cValue = interpret(c, env, args)
        if (cValue == null)
          null
        else {
          val ordering = tcoerce[TIterable](c.typ).elementType.ordering(ctx.stateManager).toOrdering
          cValue match {
            case s: Set[_] =>
              s.asInstanceOf[Set[Any]].toFastSeq.sorted(ordering)
            case d: Map[_, _] => d.iterator.map { case (k, v) => Row(k, v) }.toFastSeq.sorted(ordering)
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
              s.count(elem.typ.ordering(ctx.stateManager).lt(_, eValue))
            case d: Map[_, _] =>
              assert(onKey)
              d.count { case (k, _) => elem.typ.ordering(ctx.stateManager).lt(k, eValue) }
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
                val ordering = eltT.ordering(ctx.stateManager)
                val lb = a.count(elem => ordering.lt(eltF(elem), eValue))
                lb
              } else
                a.count(elem.typ.ordering(ctx.stateManager).lt(_, eValue))
          }
        }

      case GroupByKey(collection) =>
        interpret(collection, env, args).asInstanceOf[IndexedSeq[Row]]
          .groupBy { case Row(k, _) => k }
          .mapValues { elt: IndexedSeq[Row] => elt.map { case Row(_, v) => v } }
      case StreamTake(a, len) =>
        val aValue = interpret(a, env, args)
        val lenValue = interpret(len, env, args)
        if (aValue == null || lenValue == null)
          null
        else {
          val len = lenValue.asInstanceOf[Int]
          if (len < 0) fatal("stream take: negative num")
          aValue.asInstanceOf[IndexedSeq[Any]].take(len)
        }
      case StreamDrop(a, num) =>
        val aValue = interpret(a, env, args)
        val numValue = interpret(num, env, args)
        if (aValue == null || numValue == null)
          null
        else {
          val n = numValue.asInstanceOf[Int]
          if (n < 0) fatal("stream drop: negative num")
          aValue.asInstanceOf[IndexedSeq[Any]].drop(n)
        }
      case StreamGrouped(a, size) =>
        val aValue = interpret(a, env, args)
        val sizeValue = interpret(size, env, args)
        if (aValue == null || sizeValue == null)
          null
        else {
          val size = sizeValue.asInstanceOf[Int]
          if (size <= 0) fatal("stream grouped: non-positive size")
          aValue.asInstanceOf[IndexedSeq[Any]].grouped(size).toFastSeq
        }
      case StreamGroupByKey(a, key, missingEqual) =>
        val aValue = interpret(a, env, args)
        if (aValue == null)
          null
        else {
          val structType = tcoerce[TStruct](tcoerce[TStream](a.typ).elementType)
          val seq = aValue.asInstanceOf[IndexedSeq[Row]]
          if (seq.isEmpty)
            FastSeq[IndexedSeq[Row]]()
          else {
            val outer = new BoxedArrayBuilder[IndexedSeq[Row]]()
            val inner = new BoxedArrayBuilder[Row]()
            val (kType, getKey) = structType.select(key)
            val keyOrd = TBaseStruct.getJoinOrdering(ctx.stateManager, kType.types, missingEqual)
            var curKey: Row = getKey(seq.head)

            seq.foreach { elt =>
              val nextKey = getKey(elt)
              if (!keyOrd.equiv(curKey, nextKey)) {
                outer += inner.result()
                inner.clear()
                curKey = nextKey
              }
              inner += elt
            }
            outer += inner.result()

            outer.result().toFastSeq
          }
        }
      case StreamMap(a, name, body) =>
        val aValue = interpret(a, env, args)
        if (aValue == null)
          null
        else {
          aValue.asInstanceOf[IndexedSeq[Any]].map { element =>
            interpret(body, env.bind(name, element), args)
          }
        }
      case StreamZip(as, names, body, behavior, errorID) =>
        val aValues = as.map(interpret(_, env, args).asInstanceOf[IndexedSeq[_]])
        if (aValues.contains(null))
          null
        else {
          val len = behavior match {
            case ArrayZipBehavior.AssertSameLength | ArrayZipBehavior.AssumeSameLength =>
              val lengths = aValues.map(_.length).toSet
              if (lengths.size != 1)
                fatal(s"zip: length mismatch: ${ lengths.mkString(", ") }", errorID)
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
      case StreamMultiMerge(as, key) =>
        val streams = as.map(interpret(_, env, args).asInstanceOf[IndexedSeq[Row]])
        if (streams.contains(null))
          null
        else {
          val k = as.length
          val tournament = Array.fill[Int](k)(-1)
          val structType = tcoerce[TStruct](tcoerce[TStream](as.head.typ).elementType)
          val (kType, getKey) = structType.select(key)
          val heads = Array.fill[Int](k)(-1)
          val ordering = kType.ordering(ctx.stateManager).toOrdering.on[Row](getKey)

          def get(i: Int): Row = streams(i)(heads(i))
          def lt(li: Int, lv: Row, ri: Int, rv: Row): Boolean = {
            val c = ordering.compare(lv, rv)
            c < 0 || (c == 0 && li < ri)
          }

          def advance(i: Int) {
            heads(i) += 1
            var winner = if (heads(i) < streams(i).length) i else k
            var j = (i + k) / 2
            while (j != 0 && tournament(j) != -1) {
              val challenger = tournament(j)
              if (challenger != k && (winner == k || lt(j, get(challenger), i, get(winner)))) {
                tournament(j) = winner
                winner = challenger
              }
              j = j / 2
            }
            tournament(j) = winner
          }

          for (i <- 0 until k) { advance(i) }

          val builder = new BoxedArrayBuilder[Row]()
          while (tournament(0) != k) {
            val i = tournament(0)
            val elt = streams(i)(heads(i))
            advance(i)
            builder += elt
          }
          builder.result().toFastSeq
        }
      case StreamZipJoin(as, key, curKeyName, curValsName, joinF) =>
        val streams = as.map(interpret(_, env, args).asInstanceOf[IndexedSeq[Row]])
        if (streams.contains(null))
          null
        else {
          val k = as.length
          val tournament = Array.fill[Int](k)(-1)
          val structType = tcoerce[TStruct](tcoerce[TStream](as.head.typ).elementType)
          val (kType, getKey) = structType.select(key)
          val heads = Array.fill[Int](k)(-1)
          val ordering = kType.ordering(ctx.stateManager).toOrdering.on[Row](getKey)
          val hasKey = TBaseStruct.getJoinOrdering(ctx.stateManager, kType.types).equivNonnull _

          def get(i: Int): Row = streams(i)(heads(i))

          def advance(i: Int) {
            heads(i) += 1
            var winner = if (heads(i) < streams(i).length) i else k
            var j = (i + k) / 2
            while (j != 0 && tournament(j) != -1) {
              val challenger = tournament(j)
              if (challenger != k && (winner == k || ordering.lteq(get(challenger), get(winner)))) {
                tournament(j) = winner
                winner = challenger
              }
              j = j / 2
            }
            tournament(j) = winner
          }

          for (i <- 0 until k) { advance(i) }

          val builder = new mutable.ArrayBuffer[Any]()
          while (tournament(0) != k) {
            val i = tournament(0)
            val elt = Array.fill[Row](k)(null)
            elt(i) = streams(i)(heads(i))
            val curKey = getKey(elt(i))
            advance(i)
            var j = tournament(0)
            while (j != k && hasKey(getKey(get(j)), curKey)) {
              elt(j) = streams(j)(heads(j))
              advance(j)
              j = tournament(0)
            }
            builder += interpret(joinF, env.bind(curKeyName -> curKey, curValsName -> elt.toFastSeq), args)
          }
          builder.toFastSeq
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
      case StreamTakeWhile(a, name, cond) =>
        val aValue = interpret(a, env, args)
        if (aValue == null)
          null
        else {
          aValue.asInstanceOf[IndexedSeq[Any]].takeWhile { element =>
            // casting to boolean treats null as false
            interpret(cond, env.bind(name, element), args).asInstanceOf[Boolean]
          }
        }
      case StreamDropWhile(a, name, cond) =>
        val aValue = interpret(a, env, args)
        if (aValue == null)
          null
        else {
          aValue.asInstanceOf[IndexedSeq[Any]].dropWhile { element =>
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

      case StreamJoinRightDistinct(left, right, lKey, rKey, l, r, join, joinType) =>
        val lValue = interpret(left, env, args).asInstanceOf[IndexedSeq[Any]]
        val rValue = interpret(right, env, args).asInstanceOf[IndexedSeq[Any]]

        if (lValue == null || rValue == null)
          null
        else {
          val (lKeyTyp, lGetKey) = tcoerce[TStruct](tcoerce[TStream](left.typ).elementType).select(lKey)
          val (rKeyTyp, rGetKey) = tcoerce[TStruct](tcoerce[TStream](right.typ).elementType).select(rKey)
          assert(lKeyTyp isIsomorphicTo rKeyTyp)
          val keyOrd = TBaseStruct.getJoinOrdering(ctx.stateManager, lKeyTyp.types)

          def compF(lelt: Any, relt: Any): Int =
            keyOrd.compare(lGetKey(lelt.asInstanceOf[Row]), rGetKey(relt.asInstanceOf[Row]))
          def joinF(lelt: Any, relt: Any): Any =
            interpret(join, env.bind(l -> lelt, r -> relt), args)

          val builder = scala.collection.mutable.ArrayBuilder.make[(Option[Int], Option[Int])]
          var i = 0
          var j = 0

          while (i < lValue.length && j < rValue.length) {
            val lelt = lValue(i)
            val relt = rValue(j)
            val c = compF(lelt, relt)
            if (c < 0) {
              builder += ((Some(i), None))
              i += 1
            } else if (c > 0) {
              builder += ((None, Some(j)))
              j += 1
            } else {
              builder += ((Some(i), Some(j)))
              i += 1
              if (i == lValue.length || compF(lValue(i), relt) > 0)
                j += 1
            }
          }
          while (i < lValue.length) {
            builder += ((Some(i), None))
            i += 1
          }
          while (j < rValue.length) {
            builder += ((None, Some(j)))
            j += 1
          }

          val outerResult = builder.result()
          val elts: Iterator[(Option[Int], Option[Int])] = joinType match {
            case "inner" => outerResult.iterator.filter { case (l, r) => l.isDefined && r.isDefined }
            case "outer" => outerResult.iterator
            case "left" => outerResult.iterator.filter { case (l, r) => l.isDefined }
            case "right" => outerResult.iterator.filter { case (l, r) => r.isDefined }
          }
          elts.map { case (lIdx, rIdx) => joinF(lIdx.map(lValue.apply).orNull, rIdx.map(rValue.apply).orNull) }
            .toFastSeq
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
        val oldt = tcoerce[TStruct](old.typ)
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
      case Die(message, typ, errorId) =>
        val message_ = interpret(message).asInstanceOf[String]
        fatal(if (message_ != null) message_ else "<exception message missing>",  errorId)
      case Trap(child) =>
        try {
          Row(null, interpret(child))
        } catch {
          case e: HailException => Row(Row(e.msg, e.errorId), null)
        }
      case ConsoleLog(message, result) =>
        val message_ = interpret(message).asInstanceOf[String]
        info(message_)
        interpret(result)
      case ir@ApplyIR(function, _, _, functionArgs, _) =>
        interpret(ir.explicitNode, env, args)
      case ApplySpecial("lor", _, Seq(left_, right_), _, _) =>
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
      case ApplySpecial("land", _, Seq(left_, right_), _, _) =>
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
        val argTuple = PType.canonical(TTuple(ir.args.map(_.typ): _*)).setRequired(true).asInstanceOf[PTuple]
        ctx.r.pool.scopedRegion { region =>
          val (rt, f) = functionMemo.getOrElseUpdate(ir, {
            val wrappedArgs: IndexedSeq[BaseIR] = ir.args.zipWithIndex.map { case (x, i) =>
              GetTupleElement(Ref("in", argTuple.virtualType), i)
            }.toFastSeq
            val newChildren = ir match {
              case ir: ApplySeeded => wrappedArgs :+ NA(TRNGState)
              case _ => wrappedArgs
            }
            val wrappedIR = Copy(ir, newChildren)

            val (rt, makeFunction) = Compile[AsmFunction2RegionLongLong](ctx,
              FastSeq(("in", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(argTuple)))),
              FastSeq(classInfo[Region], LongInfo), LongInfo,
              MakeTuple.ordered(FastSeq(wrappedIR)),
              optimize = false)
            (rt.get, makeFunction(ctx.theHailClassLoader, ctx.fs, ctx.taskContext, region))
          })
          val rvb = new RegionValueBuilder(ctx.stateManager)
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
            val resultOffset = f(region, offset)
            SafeRow(rt.asInstanceOf[PTypeReferenceSingleCodeType].pt.asInstanceOf[PTuple], resultOffset).get(0)
          } catch {
            case e: Exception =>
              fatal(s"error while calling '${ ir.implementation.name }': ${ e.getMessage }", e)
          }
        }
      case TableCount(child) =>
        child.partitionCounts
          .map(_.sum)
          .getOrElse(child.analyzeAndExecute(ctx).asTableValue(ctx).rvd.count())
      case TableGetGlobals(child) =>
        child.analyzeAndExecute(ctx).asTableValue(ctx).globals.safeJavaValue
      case TableCollect(child) =>
        val tv = child.analyzeAndExecute(ctx).asTableValue(ctx)
        Row(tv.rvd.collect(ctx).toFastSeq, tv.globals.safeJavaValue)
      case TableMultiWrite(children, writer) =>
        val tvs = children.map(_.analyzeAndExecute(ctx).asTableValue(ctx))
        writer(ctx, tvs)
      case TableWrite(child, writer) =>
        writer(ctx, child.analyzeAndExecute(ctx).asTableValue(ctx))
      case BlockMatrixWrite(child, writer) =>
        writer(ctx, child.execute(ctx))
      case BlockMatrixMultiWrite(blockMatrices, writer) =>
        writer(ctx, blockMatrices.map(_.execute(ctx)))
      case TableToValueApply(child, function) =>
        function.execute(ctx, child.analyzeAndExecute(ctx).asTableValue(ctx))
      case BlockMatrixToValueApply(child, function) =>
        function.execute(ctx, child.execute(ctx))
      case BlockMatrixCollect(child) =>
        val bm = child.execute(ctx)
        // transpose because breeze toArray is column major
        val breezeMat = bm.transpose().toBreezeMatrix()
        val shape = IndexedSeq(bm.nRows, bm.nCols)
        SafeNDArray(shape, breezeMat.toArray)
      case x@TableAggregate(child, query) =>
        val value = child.analyzeAndExecute(ctx).asTableValue(ctx)
        val fsBc = ctx.fsBc

        val globalsBc = value.globals.broadcast(ctx.theHailClassLoader)
        val globalsOffset = value.globals.value.offset

        val res = genUID()

        val extracted = agg.Extract(query, res, Requiredness(x, ctx))

        val wrapped = if (extracted.aggs.isEmpty) {
          val (Some(PTypeReferenceSingleCodeType(rt: PTuple)), f) = Compile[AsmFunction2RegionLongLong](ctx,
            FastSeq(("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(value.globals.t)))),
            FastSeq(classInfo[Region], LongInfo), LongInfo,
            MakeTuple.ordered(FastSeq(extracted.postAggIR)))

          // TODO Is this right? where does wrapped run?
          ctx.scopedExecution((hcl, fs, htc, r) => SafeRow(rt, f(hcl, fs, htc, r).apply(r, globalsOffset)))
        } else {
          val spec = BufferSpec.blockedUncompressed

          val (_, initOp) = CompileWithAggregators[AsmFunction2RegionLongUnit](ctx,
            extracted.states,
            FastSeq(("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(value.globals.t)))),
            FastSeq(classInfo[Region], LongInfo), UnitInfo,
            extracted.init)

          val (_, partitionOpSeq) = CompileWithAggregators[AsmFunction3RegionLongLongUnit](ctx,
            extracted.states,
            FastSeq(("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(value.globals.t))),
              ("row", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(value.rvd.rowPType)))),
            FastSeq(classInfo[Region], LongInfo, LongInfo), UnitInfo,
            extracted.seqPerElt)

          val useTreeAggregate = extracted.shouldTreeAggregate
          val isCommutative = extracted.isCommutative
          log.info(s"Aggregate: useTreeAggregate=${ useTreeAggregate }")
          log.info(s"Aggregate: commutative=${ isCommutative }")

          // A mutable reference to a byte array. If someone higher up the
          // call stack holds a WrappedByteArray, we can set the reference
          // to null to allow the array to be GCed.
          class WrappedByteArray(_bytes: Array[Byte]) {
            private var ref: Array[Byte] = _bytes
            def bytes: Array[Byte] = ref
            def clear() { ref = null }
          }

          // creates a region, giving ownership to the caller
          val read: (HailClassLoader, HailTaskContext) => (WrappedByteArray => RegionValue) = {
            val deserialize = extracted.deserialize(ctx, spec)
            (hcl: HailClassLoader, htc: HailTaskContext) => {
              (a: WrappedByteArray) => {
                val r = Region(Region.SMALL, htc.getRegionPool())
                val res = deserialize(hcl, htc, r, a.bytes)
                a.clear()
                RegionValue(r, res)
              }
            }
          }

          // consumes a region, taking ownership from the caller
          val write: (HailClassLoader, HailTaskContext, RegionValue) => WrappedByteArray = {
            val serialize = extracted.serialize(ctx, spec)
            (hcl: HailClassLoader, htc: HailTaskContext, rv: RegionValue) => {
              val a = serialize(hcl, htc, rv.region, rv.offset)
              rv.region.invalidate()
              new WrappedByteArray(a)
            }
          }

          // takes ownership of both inputs, returns ownership of result
          val combOpF: (HailClassLoader, HailTaskContext, RegionValue, RegionValue) => RegionValue =
            extracted.combOpF(ctx, spec)

          // returns ownership of a new region holding the partition aggregation
          // result
          def itF(theHailClassLoader: HailClassLoader, i: Int, ctx: RVDContext, it: Iterator[Long]): RegionValue = {
            val partRegion = ctx.partitionRegion
            val globalsOffset = globalsBc.value.readRegionValue(partRegion, theHailClassLoader)
            val init = initOp(theHailClassLoader, fsBc.value, SparkTaskContext.get(), partRegion)
            val seqOps = partitionOpSeq(theHailClassLoader, fsBc.value, SparkTaskContext.get(), partRegion)
            val aggRegion = ctx.freshRegion(Region.SMALL)

            init.newAggState(aggRegion)
            init(partRegion, globalsOffset)
            seqOps.setAggState(aggRegion, init.getAggOffset())
            it.foreach { ptr =>
              seqOps(ctx.region, globalsOffset, ptr)
              ctx.region.clear()
            }

            RegionValue(aggRegion, seqOps.getAggOffset())
          }

          // creates a new region holding the zero value, giving ownership to
          // the caller
          val mkZero = (theHailClassLoader: HailClassLoader, tc: HailTaskContext) => {
            val region = Region(Region.SMALL, tc.getRegionPool())
            val initF = initOp(theHailClassLoader, fsBc.value, tc, region)
            initF.newAggState(region)
            initF(region, globalsBc.value.readRegionValue(region, theHailClassLoader))
            RegionValue(region, initF.getAggOffset())
          }

          val rv = value.rvd.combine[WrappedByteArray, RegionValue](
            ctx, mkZero, itF, read, write, combOpF, isCommutative, useTreeAggregate)

          val (Some(PTypeReferenceSingleCodeType(rTyp: PTuple)), f) = CompileWithAggregators[AsmFunction2RegionLongLong](
            ctx,
            extracted.states,
            FastSeq(("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(value.globals.t)))),
            FastSeq(classInfo[Region], LongInfo), LongInfo,
            Let(res, extracted.results, MakeTuple.ordered(FastSeq(extracted.postAggIR))))
          assert(rTyp.types(0).virtualType == query.typ)

          ctx.r.pool.scopedRegion { r =>
            val resF = f(ctx.theHailClassLoader, fsBc.value, ctx.taskContext, r)
            resF.setAggState(rv.region, rv.offset)
            val resAddr = resF(r, globalsOffset)
            val res = SafeRow(rTyp, resAddr)
            resF.storeAggsToRegion()
            rv.region.invalidate()
            res
          }
        }

        wrapped.get(0)
      case LiftMeOut(child) =>
        val (Some(PTypeReferenceSingleCodeType(rt)), makeFunction) = Compile[AsmFunction1RegionLong](ctx,
          FastSeq(),
          FastSeq(classInfo[Region]), LongInfo,
          MakeTuple.ordered(FastSeq(child)),
          optimize = false)
        ctx.scopedExecution { (hcl, fs, htc, r) =>
          SafeRow.read(rt, makeFunction(hcl, fs, htc, r)(r)).asInstanceOf[Row](0)
        }
      case UUID4(_) =>
         uuid4()
    }
  }
}

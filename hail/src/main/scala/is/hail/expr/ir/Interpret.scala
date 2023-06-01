package is.hail.expr.ir

import cats.data.{OptionT, StateT}
import cats.syntax.all._
import cats.{Foldable, Monad, Traverse}
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend.spark.SparkTaskContext
import is.hail.backend.{ExecuteContext, HailTaskContext}
import is.hail.expr.ir.lowering.{Lower, LoweringPipeline, MonadLower}
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
import scala.language.higherKinds
import scala.util.Try

object Interpret {
  type Agg = (IndexedSeq[Row], TStruct)

  def apply[M[_]: MonadLower](tir: TableIR, ctx: ExecuteContext): M[TableValue] =
    apply[M](tir, ctx, optimize = true)

  def apply[M[_]: MonadLower](tir: TableIR, ctx: ExecuteContext, optimize: Boolean): M[TableValue] =
    for {lowered <- LoweringPipeline.legacyRelationalLowerer(optimize)(ctx, tir)}
      yield lowered.asInstanceOf[TableIR].noSharing.analyzeAndExecute(ctx).asTableValue(ctx)

  def apply[M[_]: MonadLower](mir: MatrixIR, ctx: ExecuteContext, optimize: Boolean): M[TableValue] =
    for {lowered <- LoweringPipeline.legacyRelationalLowerer(optimize)(ctx, mir)}
      yield lowered.asInstanceOf[TableIR].analyzeAndExecute(ctx).asTableValue(ctx)

  def apply[M[_]: MonadLower](bmir: BlockMatrixIR, ctx: ExecuteContext, optimize: Boolean): M[BlockMatrix] =
    for {lowered <- LoweringPipeline.legacyRelationalLowerer(optimize)(ctx, bmir)}
      yield lowered.asInstanceOf[BlockMatrixIR].execute(ctx)

  def apply[M[_]: MonadLower, A](ctx: ExecuteContext, ir: IR)(implicit ev: Null <:< A) : M[A] =
    apply[M, A](ctx, ir, Env.empty[(Any, Type)], FastIndexedSeq[(Any, Type)]())

  def apply[M[_]: MonadLower, A](ctx: ExecuteContext,
                                 ir0: IR,
                                 env: Env[(Any, Type)],
                                 args: IndexedSeq[(Any, Type)],
                                 optimize: Boolean = true
                                 )
                                 (implicit ev: Null <:< A): M[A] =
    for {
      lowered <- LoweringPipeline.relationalLowerer(optimize)(ctx, env.m.foldLeft[IR](ir0) {
        case (acc, (k, (value, t))) => Let(k, Literal.coerce(t, value), acc)
      })
      result <- run(ctx, lowered.asInstanceOf[IR], Env.empty[Any], args, Memo.empty).value
    } yield result.map(_.asInstanceOf[A]).orNull

  def alreadyLowered[M[_]](ctx: ExecuteContext, ir: IR)(implicit M: MonadLower[M]): M[Any] =
    run(ctx, ir, Env.empty, FastIndexedSeq(), Memo.empty).getOrElseF(M.pure(null))

  private def run[M[_]](ctx: ExecuteContext,
                        ir: IR,
                        env: Env[Any],
                        args: IndexedSeq[(Any, Type)],
                        functionMemo: Memo[(SingleCodeType, AsmFunction2RegionLongLong)]
                       )
                       (implicit M: MonadLower[M])
  : OptionT[M, Any] = {

    type F[A] = OptionT[M, A]
    val F = Monad[F]

    def empty = OptionT.none[M, Any]

    def interpret(ir: IR, env: Env[Any] = env, args: IndexedSeq[(Any, Type)] = args): F[Any] =
      run(ctx, ir, env, args, functionMemo)

    ir match {
      case I32(x) => F.pure(x)
      case I64(x) => F.pure(x)
      case F32(x) => F.pure(x)
      case F64(x) => F.pure(x)
      case Str(x) => F.pure(x)
      case True() => F.pure(true)
      case False() => F.pure(false)
      case Literal(_, value) => F.pure(value)
      case x@EncodedLiteral(codec, value) =>
        F.pure {
          ctx.r.getPool().scopedRegion { r =>
            val (pt, addr) = codec.decodeArrays(ctx, x.typ, value.ba, ctx.r)
            SafeRow.read(pt, addr)
          }
        }
      case Void() => F.unit.widen[Any]
      case Cast(v, t) =>
        interpret(v, env, args).map { vValue =>
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
        }

      case CastRename(v, _) => interpret(v)
      case NA(_) => empty

      case IsNA(value) => interpret(value, env, args).map(_ == null)
      case Coalesce(values) =>
        FastSeq(values: _*).foldMapK { x =>
          interpret(x, env, args)
        }

      case If(cond, cnsq, altr) =>
        assert(cnsq.typ == altr.typ)
        interpret(cond, env, args).flatMap { case tf: Boolean =>
          if (tf) interpret(cnsq, env, args)
          else interpret(altr, env, args)
        }

      case Let(name, value, body) =>
        for {
          v <- OptionT.liftF(interpret(value, env, args).value)
          r <- interpret(body, env.bind(name, v.orNull), args)
        } yield r

      case Ref(name, _) =>
        F.pure(env.lookup(name))

      case ApplyBinaryPrimOp(op, l, r) =>
        for {
          lValue <- interpret(l, env, args)
          rValue <- interpret(r, env, args)
        } yield (l.typ, r.typ) match {
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
        for {xValue <- interpret(x, env, args)}
          yield op match {
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
            assert(x.typ.isInstanceOf[TIntegral])
            x.typ match {
              case TInt32 => ~xValue.asInstanceOf[Int]
              case TInt64 => ~xValue.asInstanceOf[Long]
            }
          case BitCount() =>
            assert(x.typ.isInstanceOf[TIntegral])
            x.typ match {
              case TInt32 => Integer.bitCount(xValue.asInstanceOf[Int])
              case TInt64 => java.lang.Long.bitCount(xValue.asInstanceOf[Long])
            }
        }

      case ApplyComparisonOp(op, l, r) =>
        OptionT {
          for {
            lValue <- interpret(l, env, args).value
            rValue <- interpret(r, env, args).value
          } yield Some(op).filter(_.strict && (lValue == null || rValue == null)).map {
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
        }

      case MakeArray(elements, _) =>
        elements.traverse(interpret(_, env, args).orElse(F.pure(null))).widen[Any]

      case MakeStream(elements, _, _) =>
        elements.traverse(interpret(_, env, args).orElse(F.pure(null))).widen[Any]

      case x@ArrayRef(a, i, errorId) =>
        for {
          aValue <- interpret(a, env, args)
          iValue <- interpret(i, env, args)
        } yield {
          val a = aValue.asInstanceOf[IndexedSeq[Any]]
          val i = iValue.asInstanceOf[Int]

          if (i < 0 || i >= a.length) {
            fatal(s"array index out of bounds: index=$i, length=${ a.length }", errorId = errorId)
          } else
            a.apply(i)
        }

      case ArraySlice(a, start, stop, step, errorID) =>
        for {
          aValue <- interpret(a, env, args)
          a = aValue.asInstanceOf[IndexedSeq[Any]]

          startValue <- interpret(start, env, args)
          stopValue <- stop.traverse(interpret(_, env, args))
          if stopValue.getOrElse(a.size) != null
          stepValue <- interpret(step, env, args)
        } yield {
          val requestedStart = startValue.asInstanceOf[Int]
          val requestedStep = stepValue.asInstanceOf[Int]

          if (requestedStep == 0)
            fatal("step cannot be 0 for array slice", errorID)

          val noneStop = if (requestedStep < 0) -a.size - 1 else a.size
          val maxBound = if (requestedStep > 0) a.size else a.size - 1
          val minBound = if (requestedStep > 0) 0 else -1
          val requestedStop = stopValue.getOrElse(noneStop).asInstanceOf[Int]

          val realStart =
            if (requestedStart >= a.size) maxBound
            else if (requestedStart >= 0) requestedStart
            else if (requestedStart + a.size >= 0) requestedStart + a.size
            else minBound

          val realStop =
            if (requestedStop >= a.size) maxBound
            else if (requestedStop >= 0) requestedStop
            else if (requestedStop + a.size > 0) requestedStop + a.size
            else minBound

          (realStart until realStop by requestedStep).map(idx => a(idx))
        }

      case ArrayLen(a) =>
        for {aValue <- interpret(a, env, args)}
          yield aValue.asInstanceOf[IndexedSeq[Any]].length

      case StreamLen(a) => for {aValue <- interpret(a, env, args)}
        yield aValue.asInstanceOf[IndexedSeq[Any]].length

      case _: StreamIota =>
        throw new UnsupportedOperationException

      case StreamRange(start, stop, step, _, errorID) =>
        for {
          startValue <- interpret(start, env, args)
          stopValue <- interpret(stop, env, args)
          stepValue <- interpret(step, env, args)
          _ <- F.whenA(stepValue == 0) { fatal("Array range cannot have step size 0.", errorID) }
        } yield startValue.asInstanceOf[Int] until stopValue.asInstanceOf[Int] by stepValue.asInstanceOf[Int]

      case ArraySort(a, l, r, lessThan) =>
        interpret(a, env, args).flatMap { case aValue: IndexedSeq[Any] =>
          OptionT.liftF {
            M.lift {
              Lower { (ctx, s0) =>
                var s = s0
                val sorted = Try {
                  aValue.sortWith { (left, right) =>
                    if (left != null && right != null) {
                      val (s_, res) =
                        run[Lower](ctx, lessThan, env.bind(l, left).bind(r, right), args, functionMemo)
                          .value
                          .run(ctx, s)

                      s = s_

                      res match {
                        case Left(t) => throw t
                        case Right(r) =>
                          if (r.isEmpty || r.contains(null)) fatal ("Result of sorting function cannot be missing.")
                          else r.get.asInstanceOf[Boolean]
                      }
                    }
                    else right == null
                  }
                }
                (s, sorted.toEither)
              }
            }
          }
        }

      case ToSet(a) =>
        for {aValue <- interpret(a, env, args)}
          yield aValue.asInstanceOf[IndexedSeq[Any]].toSet

      case ToDict(a) =>
        for {aValue <- interpret(a, env, args)}
          yield aValue.asInstanceOf[IndexedSeq[Any]].filter(_ != null).map { case Row(k, v) => (k, v) }.toMap

      case _: CastToArray | _: ToArray | _: ToStream =>
        val c = ir.children(0).asInstanceOf[IR]
        for { cValue <- interpret(c, env, args) }
          yield {
            val ordering = tcoerce[TIterable](c.typ).elementType.ordering(ctx.stateManager).toOrdering
            cValue match {
              case s: Set[_] =>
                s.asInstanceOf[Set[Any]].toFastIndexedSeq.sorted(ordering)
              case d: Map[_, _] => d.iterator.map { case (k, v) => Row(k, v) }.toFastIndexedSeq.sorted(ordering)
              case a => a
            }
          }

      case LowerBoundOnOrderedCollection(orderedCollection, elem, onKey) =>
        for {
          cValue <- interpret(orderedCollection, env, args)
          eValue <- OptionT.liftF(interpret(elem, env, args).value)
        } yield {
          cValue match {
            case s: Set[_] =>
              assert(!onKey)
              s.count(elem.typ.ordering(ctx.stateManager).lt(_, eValue.orNull))

            case d: Map[_, _] =>
              assert(onKey)
              d.count { case (k, _) => elem.typ.ordering(ctx.stateManager).lt(k, eValue.orNull) }

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
                val lb = a.count(elem => ordering.lt(eltF(elem), eValue.orNull))
                lb
              } else
                a.count(elem.typ.ordering(ctx.stateManager).lt(_, eValue.orNull))
          }
        }

      case GroupByKey(collection) =>
        for {c <- interpret(collection, env, args)}
          yield c.asInstanceOf[IndexedSeq[Row]]
            .groupBy { case Row(k, _) => k }
            .mapValues { elt: IndexedSeq[Row] => elt.map { case Row(_, v) => v } }

      case StreamTake(a, len) =>
        for {
          aValue <- interpret(a, env, args)
          lenValue <- interpret(len, env, args)
        } yield {
          val len = lenValue.asInstanceOf[Int]
          if (len < 0) fatal("stream take: negative num")
          aValue.asInstanceOf[IndexedSeq[Any]].take(len)
        }

      case StreamDrop(a, num) =>
        for {
          aValue <- interpret(a, env, args)
          numValue <- interpret(num, env, args)
        } yield {
          val n = numValue.asInstanceOf[Int]
          if (n < 0) fatal("stream drop: negative num")
          aValue.asInstanceOf[IndexedSeq[Any]].drop(n)
        }

      case StreamGrouped(a, size) =>
        for {
          aValue <- interpret(a, env, args)
          sizeValue <- interpret(size, env, args)
        } yield {
          val size = sizeValue.asInstanceOf[Int]
          if (size <= 0) fatal("stream grouped: non-positive size")
          aValue.asInstanceOf[IndexedSeq[Any]].grouped(size).toFastIndexedSeq
        }

      case StreamGroupByKey(a, key, missingEqual) =>
        for {aValue <- interpret(a, env, args)}
          yield {
            val structType = tcoerce[TStruct](tcoerce[TStream](a.typ).elementType)
            val seq = aValue.asInstanceOf[IndexedSeq[Row]]
            if (seq.isEmpty)
              FastIndexedSeq[IndexedSeq[Row]]()
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

              outer.result().toFastIndexedSeq
            }
          }

      case StreamMap(a, name, body) =>
        for {
          aValue <- interpret(a, env, args)
          mapd <- aValue.asInstanceOf[IndexedSeq[Any]].traverse { x =>
            interpret(body, env.bind(name, x), args).orElse(F.pure(null))
          }
        } yield mapd

      case StreamZip(as, names, body, behavior, errorID) =>
        for {
          aValues <- as.traverse { a =>
            interpret(a, env, args).map(_.asInstanceOf[mutable.IndexedSeq[Any]])
          }

          len = behavior match {
            case ArrayZipBehavior.AssertSameLength | ArrayZipBehavior.AssumeSameLength =>
              val lengths = aValues.map(_.length).toSet
              if (lengths.size != 1)
                fatal(s"zip: length mismatch: ${lengths.mkString(", ")}", errorID)
              lengths.head
            case ArrayZipBehavior.TakeMinLength =>
              aValues.map(_.length).min
            case ArrayZipBehavior.ExtendNA =>
              aValues.map(_.length).max
          }

          res <- Traverse[IndexedSeq].traverse(0 until len) { i =>
            val e = env.bindIterable(names.zip(aValues.map(a => if (i >= a.length) null else a.apply(i))))
            interpret(body, e, args).orElse(F.pure(null))
          }

        } yield res

      case StreamMultiMerge(as, key) =>
        for {streams <- as.traverse(interpret(_, env, args).map(_.asInstanceOf[IndexedSeq[Row]]))}
          yield {
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

            for (i <- 0 until k) {
              advance(i)
            }

            val builder = new BoxedArrayBuilder[Row]()
            while (tournament(0) != k) {
              val i = tournament(0)
              val elt = streams(i)(heads(i))
              advance(i)
              builder += elt
            }
            builder.result().toFastIndexedSeq
          }

      case StreamZipJoin(as, key, curKeyName, curValsName, joinF) =>
        for {streams <- as.traverse(interpret(_, env, args).map(_.asInstanceOf[IndexedSeq[Row]]))}
          yield {
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

            for (i <- 0 until k) {
              advance(i)
            }

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
              builder += interpret(joinF, env.bind(curKeyName -> curKey, curValsName -> elt.toFastIndexedSeq), args)
            }
            builder.toFastIndexedSeq
          }

      case StreamFilter(a, name, cond) =>
        for {
          aValue <- interpret(a, env, args)
          filterd <- aValue.asInstanceOf[IndexedSeq[Any]].filterA { x =>
            interpret(cond, env.bind(name, x), args).map(_.asInstanceOf[Boolean]).orElse(F.pure(false))
          }
        } yield filterd

      case StreamTakeWhile(a, name, cond) =>
        interpret(a, env, args).flatMap { case as: IndexedSeq[Any] =>
          (for {idx <- StateT.get[F, Int]; _ <- StateT.set[F, Int](idx + 1)} yield as(idx))
            .whileM[Vector] {
              StateT.get[F, Int].flatMap { idx =>
                if (idx >= as.length) StateT.pure(false)
                else StateT.liftF {
                  interpret(cond, env.bind(name, as(idx)), args)
                    .map(_.asInstanceOf[Boolean])
                    .orElse(F.pure(false))
                }
              }
            }
            .runA(0)
            .widen[Any]
        }

      case StreamDropWhile(a, name, cond) =>
        for {
          aValue <- interpret(a, env, args)
          as = aValue.asInstanceOf[IndexedSeq[Any]]
          first <-
            (for {idx <- StateT.get[F, Int]; _ <- StateT.set[F, Int](idx + 1)} yield as(idx))
              .whileM_ {
                StateT.get[F, Int].flatMap { idx =>
                  if (idx >= as.length) StateT.pure(false)
                  else StateT.liftF {
                    interpret(cond, env.bind(name, as(idx)), args)
                      .map(_.asInstanceOf[Boolean])
                      .orElse(F.pure(false))
                  }
                }
              }
              .runS(0)
        } yield as.drop(first)

      case StreamFlatMap(a, name, body) =>
        for {
          as <- interpret(a, env, args)
          bs <- as.asInstanceOf[IndexedSeq[Any]].foldMapK { a =>
            interpret(body, env.bind(name, a), args)
              .map(_.asInstanceOf[IndexedSeq[Any]])
              .orElse(F.pure(IndexedSeq.empty[Any]))
          }
        } yield bs

      case StreamFold(a, zero, accumName, valueName, body) =>
        for {
          aValue <- interpret(a, env, args)
          zeroValue <- interpret(zero, env, args)
          result <- aValue.asInstanceOf[IndexedSeq[Any]].foldM(zeroValue) { (accum, element) =>
            interpret(body, env.bind(accumName -> accum, valueName -> element), args).orElse(F.pure(null))
          }
        } yield result

      case StreamFold2(a, accum, valueName, seq, res) =>
        for {
          aValue <- interpret(a, env, args)
          accVals <- accum.traverse { case (name, value) =>
            interpret(value, env, args)
              .orElse(F.pure(null))
              .map((name, _))
          }

          indices = accVals.asInstanceOf[IndexedSeq[Any]].indices
          e <- aValue.asInstanceOf[IndexedSeq[Any]].foldM(env.bindIterable(accVals)) { case (e, elt) =>
            Foldable[IndexedSeq].foldM(indices, e.bind(valueName, elt)) { case (e, i) =>
              for {v <- interpret(seq(i), e, args).orElse(F.pure(null))}
                yield e.bind(accum(i)._1, v)
            }
          }

          result <- interpret(res, e.delete(valueName), args)
        } yield result

      case StreamScan(a, zero, accumName, valueName, body) =>
        for {
          aValue <- interpret(a, env, args)
          zeroValue <- interpret(zero, env, args)
          result <- aValue.asInstanceOf[IndexedSeq[Any]].foldM(Vector(zeroValue)) { (accum, elt) =>
            interpret(body, env.bind(accumName -> accum.last, valueName -> elt), args)
              .orElse(F.pure(null))
              .map(b => accum :+ b)
          }
        } yield result

      case StreamJoinRightDistinct(left, right, lKey, rKey, l, r, join, joinType) =>
        for {
          lValue <- interpret(left, env, args).map(_.asInstanceOf[IndexedSeq[Any]])
          rValue <- interpret(right, env, args).map(_.asInstanceOf[IndexedSeq[Any]])
        } yield {
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
            .toFastIndexedSeq
        }

      case StreamFor(a, valueName, body) =>
        for {
          aValue <- interpret(a, env, args)
          _ <- aValue.asInstanceOf[IndexedSeq[Any]].traverse_ { element =>
            interpret(body, env.bind(valueName -> element), args).void.orElse(F.unit)
          }
        } yield ().asInstanceOf[Any]

      case Begin(xs) =>
        xs.traverse_(interpret(_)).widen[Any]


      case MakeStruct(fields) =>
        F.pure(Row.fromSeq(_)) ap fields.traverse { case (_, fieldIR) =>
          interpret(fieldIR, env, args).orElse(F.pure(null))
        }

      case SelectFields(old, fields) =>
        interpret(old, env, args).map { case oldRow: Row =>
          val oldt = tcoerce[TStruct](old.typ)
          Row.fromSeq(fields.map(id => oldRow.get(oldt.fieldIdx(id))))
        }

      case x@InsertFields(old, fields, fieldOrder) =>
        interpret(old, env, args).map { case struct: Row =>
          fieldOrder match {
            case Some(fds) =>
              FastSeq(fields: _*).traverse { case (name, ir) =>
                F.pure((name, _: Any)) ap interpret(ir, env, args)
              }
                .map(_.toMap)
                .map { newValues =>
                  val oldIndices = old.typ.asInstanceOf[TStruct].fields.map(f => f.name -> f.index).toMap
                  Row.fromSeq(fds.map(name => newValues.getOrElse(name, struct.get(oldIndices(name)))))
                }

            case None =>
              FastSeq(fields: _*).foldM((old.typ.asInstanceOf[TStruct], struct.asInstanceOf[Any])) {
                case ((t, struct), (name, body)) =>
                  interpret(body, env, args).map { v =>
                    val (newT, ins) = t.insert(body.typ, name)
                    (newT.asInstanceOf[TStruct], ins(struct, v))
                  }
              }
          }
        }

      case GetField(o, name) =>
        interpret(o, env, args).map { case oValue: Row =>
          val oType = o.typ.asInstanceOf[TStruct]
          val fieldIndex = oType.fieldIdx(name)
          oValue.get(fieldIndex)
        }

      case MakeTuple(types) =>
        F.pure(Row.fromSeq(_)) ap types.traverse { case (_, x) =>
          interpret(x, env, args).orElse(F.pure(null))
        }

      case GetTupleElement(o, idx) =>
        interpret(o, env, args).map { case r: Row =>
          r.get(o.typ.asInstanceOf[TTuple].fieldIndex(idx))
        }

      case In(i, _) =>
        val (a, _) = args(i)
        F.pure(a)

      case Die(message, _, errorId) =>
        val message_ = interpret(message).asInstanceOf[String]
        fatal(if (message_ != null) message_ else "<exception message missing>",  errorId)

      case Trap(child) =>
        interpret(child).map(Row(null, _)).widen[Any].handleError {
          case e: HailException =>
            Row(Row(e.msg, e.errorId), null)
          case t =>
            throw t
        }

      case ConsoleLog(message, result) =>
        for {
          message_ <- interpret(message)
          _ = info(message_.asInstanceOf[String])
          r <- interpret(result)
        } yield r

      case ir: ApplyIR =>
        interpret(ir.explicitNode, env, args)

      case ApplySpecial("lor", _, Seq(left_, right_), _, _) =>
        interpret(left_).map(_.asInstanceOf[Boolean]).ifF(F.pure(true), interpret(right_))

      case ApplySpecial("land", _, Seq(left_, right_), _, _) =>
        interpret(left_).map(_.asInstanceOf[Boolean]).ifF(interpret(right_), F.pure(false))

      case ir: AbstractApplyNode[_] =>
        OptionT.liftF {
          M.lift {
            Lower { (ctx, s0) =>
              var state = s0
              val result = Try {
                val argTuple = PType.canonical(TTuple(ir.args.map(_.typ): _*)).setRequired(true).asInstanceOf[PTuple]
                ctx.r.pool.scopedRegion { region =>
                  val (rt, f) = functionMemo.getOrElseUpdate(ir, {
                    val wrappedArgs = ir.args.toFastIndexedSeq.zipWithIndex.map { case (x, i) =>
                      GetTupleElement(Ref("in", argTuple.virtualType), i)
                    }

                    val wrappedIR = Copy(ir, ir match {
                      case _: ApplySeeded => wrappedArgs :+ NA(TRNGState)
                      case _ => wrappedArgs
                    })

                    val (s1, r) =
                      Compile[Lower, AsmFunction2RegionLongLong](ctx,
                        FastIndexedSeq(("in", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(argTuple)))),
                        FastIndexedSeq(classInfo[Region], LongInfo), LongInfo,
                        MakeTuple.ordered(FastSeq(wrappedIR)),
                        optimize = false
                      )
                        .run(ctx, s0)
                    state = s1
                    r match {
                      case Left(t) => throw t
                      case Right((rt, makeFunction)) =>
                        (rt.get, makeFunction(ctx.theHailClassLoader, ctx.fs, ctx.taskContext, region))
                    }
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
                    (state, SafeRow(rt.asInstanceOf[PTypeReferenceSingleCodeType].pt.asInstanceOf[PTuple], resultOffset).get(0))
                  } catch {
                    case e: Exception =>
                      fatal(s"error while calling '${ir.implementation.name}': ${e.getMessage}", e)
                  }
                }
              }

              (state, result.toEither)
            }
          }
        }

      case TableCount(child) =>
        F.pure {
          child.partitionCounts
            .map(_.sum)
            .getOrElse(child.analyzeAndExecute(ctx).asTableValue(ctx).rvd.count())
        }

      case TableGetGlobals(child) =>
        F.pure {
          child.analyzeAndExecute(ctx).asTableValue(ctx).globals.safeJavaValue
        }

      case TableCollect(child) =>
        F.pure {
          val tv = child.analyzeAndExecute(ctx).asTableValue(ctx)
          Row(tv.rvd.collect(ctx).toFastIndexedSeq, tv.globals.safeJavaValue)
        }

      case TableMultiWrite(children, writer) =>
        F.pure {
          val tvs = children.map(_.analyzeAndExecute(ctx).asTableValue(ctx))
          writer(ctx, tvs)
        }

      case TableWrite(child, writer) =>
        F.pure {
          writer(ctx, child.analyzeAndExecute(ctx).asTableValue(ctx))
        }

      case BlockMatrixWrite(child, writer) =>
        F.pure {
          writer(ctx, child.execute(ctx))
        }

      case BlockMatrixMultiWrite(blockMatrices, writer) =>
        F.pure {
          writer(ctx, blockMatrices.map(_.execute(ctx)))
        }

      case TableToValueApply(child, function) =>
        F.pure {
          function.execute(ctx, child.analyzeAndExecute(ctx).asTableValue(ctx))
        }

      case BlockMatrixToValueApply(child, function) =>
        F.pure {
          function.execute(ctx, child.execute(ctx))
        }

      case BlockMatrixCollect(child) =>
        F.pure {
          val bm = child.execute(ctx)
          // transpose because breeze toArray is column major
          val breezeMat = bm.transpose().toBreezeMatrix()
          val shape = IndexedSeq(bm.nRows, bm.nCols)
          SafeNDArray(shape, breezeMat.toArray)
        }

      case x@TableAggregate(child, query) =>
        val value = child.analyzeAndExecute(ctx).asTableValue(ctx)
        val fsBc = ctx.fsBc

        val globalsBc = value.globals.broadcast(ctx.theHailClassLoader)
        val globalsOffset = value.globals.value.offset

        val res = genUID()

        val extracted = agg.Extract(query, res, Requiredness(x, ctx))

         val wrapped: M[Row] = if (extracted.aggs.isEmpty) {
          for {
            (Some(PTypeReferenceSingleCodeType(rt: PTuple)), f) <-
              Compile[M, AsmFunction2RegionLongLong](ctx,
                FastIndexedSeq(("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(value.globals.t)))),
                FastIndexedSeq(classInfo[Region], LongInfo), LongInfo,
                MakeTuple.ordered(FastSeq(extracted.postAggIR))
              )
            // TODO Is this right? where does wrapped run?
          } yield ctx.scopedExecution((hcl, fs, htc, r) => SafeRow(rt, f(hcl, fs, htc, r)(r, globalsOffset)))
        } else {
          // A mutable reference to a byte array. If someone higher up the
          // call stack holds a WrappedByteArray, we can set the reference
          // to null to allow the array to be GCed.
          class WrappedByteArray(_bytes: Array[Byte]) {
            private var ref: Array[Byte] = _bytes
            def bytes: Array[Byte] = ref
            def clear(): Unit = {
              ref = null
            }
          }

          for {
            (_, initOp) <-
              CompileWithAggregators[M, AsmFunction2RegionLongUnit](ctx,
                extracted.states,
                FastIndexedSeq(("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(value.globals.t)))),
                FastIndexedSeq(classInfo[Region], LongInfo), UnitInfo,
                extracted.init
              )

            (_, partitionOpSeq) <-
              CompileWithAggregators[M, AsmFunction3RegionLongLongUnit](ctx,
                extracted.states,
                FastIndexedSeq(("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(value.globals.t))),
                  ("row", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(value.rvd.rowPType)))),
                FastIndexedSeq(classInfo[Region], LongInfo, LongInfo), UnitInfo,
                extracted.seqPerElt
              )

            useTreeAggregate = extracted.shouldTreeAggregate
            isCommutative = extracted.isCommutative
            _ = log.info(s"Aggregate: useTreeAggregate=${useTreeAggregate}")
            _ = log.info(s"Aggregate: commutative=${isCommutative}")


            spec = BufferSpec.defaultUncompressed

            // creates a region, giving ownership to the caller
            read <- extracted.deserialize(ctx, spec).map { deserialize =>
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
            write <- extracted.serialize(ctx, spec).map { serialize =>
              (hcl: HailClassLoader, htc: HailTaskContext, rv: RegionValue) => {
                val a = serialize(hcl, htc, rv.region, rv.offset)
                rv.region.invalidate()
                new WrappedByteArray(a)
              }
            }

            // takes ownership of both inputs, returns ownership of result
            combOpF = extracted.combOpF(ctx, spec)

            // returns ownership of a new region holding the partition aggregation
            // result
            itF = (theHailClassLoader: HailClassLoader, i: Int, ctx: RVDContext, it: Iterator[Long]) => {
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
            mkZero = (theHailClassLoader: HailClassLoader, tc: HailTaskContext) => {
              val region = Region(Region.SMALL, tc.getRegionPool())
              val initF = initOp(theHailClassLoader, fsBc.value, tc, region)
              initF.newAggState(region)
              initF(region, globalsBc.value.readRegionValue(region, theHailClassLoader))
              RegionValue(region, initF.getAggOffset())
            }

            rv = value.rvd.combine[WrappedByteArray, RegionValue](ctx, mkZero, itF, read, write,
              combOpF, isCommutative, useTreeAggregate
            )

            (Some(PTypeReferenceSingleCodeType(rTyp: PTuple)), f) <-
              CompileWithAggregators[M, AsmFunction2RegionLongLong](
                ctx,
                extracted.states,
                FastIndexedSeq(("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(value.globals.t)))),
                FastIndexedSeq(classInfo[Region], LongInfo), LongInfo,
                Let(res, extracted.results, MakeTuple.ordered(FastSeq(extracted.postAggIR)))
              )

            _ = assert(rTyp.types(0).virtualType == query.typ)

          } yield ctx.r.pool.scopedRegion { r =>
            val resF = f(ctx.theHailClassLoader, fsBc.value, ctx.taskContext, r)
            resF.setAggState(rv.region, rv.offset)
            val resAddr = resF(r, globalsOffset)
            val res = SafeRow(rTyp, resAddr)
            resF.storeAggsToRegion()
            rv.region.invalidate()
            res
          }
        }

        OptionT.liftF(wrapped.map(_.get(0)))

      case LiftMeOut(child) =>
        OptionT.liftF {
          for {
            (Some(PTypeReferenceSingleCodeType(rt)), makeFunction) <-
              Compile[M, AsmFunction1RegionLong](ctx,
                FastIndexedSeq(),
                FastIndexedSeq(classInfo[Region]), LongInfo,
                MakeTuple.ordered(FastSeq(child)),
                optimize = false
              )
          } yield ctx.scopedExecution { (hcl, fs, htc, r) =>
            SafeRow.read(rt, makeFunction(hcl, fs, htc, r)(r)).asInstanceOf[Row](0)
          }
        }

      case UUID4(_) =>
         F.pure(uuid4())
    }
  }
}

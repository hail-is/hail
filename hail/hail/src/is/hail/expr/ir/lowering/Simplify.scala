package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.collection.implicits.{toRichIndexedSeq, toRichMap}
import is.hail.expr.ir.{Memoized => M, _}
import is.hail.expr.ir.Scope.EVAL
import is.hail.expr.ir.analyses.{ColumnCount, PartitionCounts, PartitionCountsOrColumnCount}
import is.hail.expr.ir.defs._
import is.hail.io.bgen.MatrixBGENReader
import is.hail.rvd.PartitionBoundOrdering
import is.hail.types.tcoerce
import is.hail.types.virtual._
import is.hail.utils._

import scala.collection.immutable.HashSet
import scala.collection.mutable

object Simplify {

  // Transform 'ir' using simplification rules until none apply.
  def apply(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    TimedBlock.enter(recur(ctx, ir))

  private[this] def recur(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    ir match {
      case ir: IR => if (IsConstant(ir)) ir else simplifyValue(ctx, ir)
      case tir: TableIR => simplifyTable(ctx, tir)
      case mir: MatrixIR => simplifyMatrix(ctx, mir)
      case bmir: BlockMatrixIR => simplifyBlockMatrix(ctx, bmir)
    }

  private[this] def visitNode[T <: BaseIR](
    transform: (ExecuteContext, T) => Option[T],
    post: => (ExecuteContext, T) => T,
  )(
    ctx: ExecuteContext,
    t: T,
  ): T = {
    val t1 = t.mapChildren(recur(ctx, _)).asInstanceOf[T]
    transform(ctx, t1) map { post(ctx, _) } getOrElse t1
  }

  private[this] val simplifyValue: (ExecuteContext, IR) => IR =
    visitNode(valueRules, simplifyValue)

  private[this] val simplifyTable: (ExecuteContext, TableIR) => TableIR =
    visitNode(tableRules, simplifyTable)

  private[this] val simplifyMatrix: (ExecuteContext, MatrixIR) => MatrixIR =
    visitNode(matrixRules, simplifyMatrix)

  private[this] val simplifyBlockMatrix: (ExecuteContext, BlockMatrixIR) => BlockMatrixIR =
    visitNode(blockMatrixRules, simplifyBlockMatrix)

  private[this] def valueRules(ctx: ExecuteContext, ir: IR): Option[IR] =
    ir match {
      // propagate NA
      case x: IR if hasMissingStrictChild(x) =>
        Some(NA(x.typ))

      case _: Coalesce
          | _: IsNA
          | _: If
          | _: Switch =>
        conditionalRules(ir)

      case _: ApplyIR |
          _: ApplyBinaryPrimOp |
          _: ApplyUnaryPrimOp |
          _: ApplyComparisonOp |
          _: ApplySpecial |
          _: BlockMatrixToValueApply =>
        applyRules(ir)

      case _: ArrayLen |
          _: ArraySlice |
          _: ArrayRef =>
        arrayRules(ir)

      case _: StreamAgg |
          _: StreamAggScan |
          _: StreamFlatMap |
          _: StreamFold |
          _: StreamFor |
          _: StreamFilter |
          _: StreamLen |
          _: StreamMap |
          _: StreamRange |
          _: StreamZip =>
        streamRules(ir)

      case _: StreamTake |
          _: StreamTakeWhile |
          _: StreamDrop |
          _: StreamDropWhile =>
        streamSubsetRules(ir)

      case _: Cast |
          _: CastRename |
          _: CastToArray |
          _: ToArray |
          _: ToStream =>
        castRules(ir)

      case _: MakeNDArray |
          _: NDArrayShape |
          _: NDArrayMap =>
        ndarrayRules(ir)

      case _: GetField |
          _: InsertFields =>
        structRules(ir)

      case _: Block =>
        blockRules(ir)

      case _: SelectFields |
          _: MakeStruct |
          _: GetTupleElement =>
        selectRules(ir)

      case _: TableCount |
          _: MatrixCount =>
        countRules(ir)

      case _: TableGetGlobals |
          _: TableCollect |
          _: TableAggregate =>
        tableValueRules(ir)

      case _ =>
        None
    }

  // Returns true if any strict child of 'x' is NA. A child is strict if 'x' evaluates to missing
  // whenever the child does.
  private[this] def hasMissingStrictChild(x: IR): Boolean = {
    x match {
      case _: Apply |
          _: ApplyUnaryPrimOp |
          _: ApplyBinaryPrimOp |
          _: ArrayRef |
          _: ArrayLen |
          _: GetField |
          _: GetTupleElement => x.children.exists(_.isInstanceOf[NA])
      case ApplyComparisonOp(op, _, _) if op.strict => x.children.exists(_.isInstanceOf[NA])
      case _ => false
    }
  }

  // bound only by MatrixIR and TableIR, references to which cannot be missing
  private[this] val nonMissingNames =
    HashSet(
      MatrixIR.globalName,
      MatrixIR.rowName,
      MatrixIR.colName,
      TableIR.globalName,
      TableIR.rowName,
    )

  // Returns true if 'x' will never evaluate to missing.
  private[this] def isDefinitelyDefined(x: IR): Boolean =
    x match {
      case _: MakeArray |
          _: MakeStream |
          _: MakeStruct |
          _: MakeTuple |
          _: IsNA |
          ApplyComparisonOp(EQWithNA, _, _) |
          ApplyComparisonOp(NEQWithNA, _, _) |
          _: I32 | _: I64 | _: F32 | _: F64 | True() | False() =>
        true

      case Ref(name, _) =>
        nonMissingNames contains name

      case _: StreamDrop |
          _: StreamTake |
          _: StreamRange =>
        x.children.forall { case ir: IR => isDefinitelyDefined(ir) }

      case _ => false
    }

  private[this] def binaryPrimOpRules(ir: IR): Option[IR] =
    if (!ir.typ.isPrimitive) None
    else {
      val typ = ir.typ

      def isIntegral(t: Type) = t.isInstanceOf[TIntegral]

      def pure(x: Int) = Literal.coerce(typ, x)
      @inline def MinusOne = pure(-1)
      @inline def Zero = pure(0)
      @inline def One = pure(1)
      @inline def Two = pure(2)

      ir match {
        case ApplyBinaryPrimOp(op, x, y) =>
          op match {
            case Add() =>
              val `0` = Zero
              if (x == `0`) Some(y)
              else if (y == `0`) Some(x)
              else if (isIntegral(typ) && x == y) Some(ApplyBinaryPrimOp(Multiply(), Two, x))
              else None

            case Subtract() =>
              val `0` = Zero
              if (x == `0`) Some(ApplyUnaryPrimOp(Negate, y))
              else if (y == `0`) Some(x)
              else if (isIntegral(typ) && x == y) Some(`0`)
              else None

            case Multiply() =>
              val `1` = One
              if (x == `1`) Some(y)
              else {
                val `-1` = MinusOne
                if (x == `-1`) Some(ApplyUnaryPrimOp(Negate, y))
                else if (y == `1`) Some(x)
                else if (y == `-1`) Some(ApplyUnaryPrimOp(Negate, x))
                else {
                  val `0` = Zero
                  if (isIntegral(typ) && (x == `0` || y == `0`)) Some(`0`)
                  else None
                }
              }

            case RoundToNegInfDivide() =>
              val `1` = One
              if (y == `1`) Some(x)
              else if (y == MinusOne) Some(ApplyUnaryPrimOp(Negate, x))
              else if (isIntegral(typ)) {
                val `0` = Zero
                if (x == y) Some(`1`)
                else if (x == `0`) Some(x)
                else if (y == `0`) Some(Die("division by zero", ir.typ))
                else None
              } else None

            case _: LeftShift | _: RightShift | _: LogicalRightShift if isIntegral(typ) =>
              if (x == Zero || y == I32(0)) Some(x)
              else None

            case BitAnd() if isIntegral(typ) =>
              val `0` = Zero
              if (x == `0` || y == `0`) Some(`0`)
              else {
                val `-1` = MinusOne
                if (x == `-1`) Some(y)
                else if (y == `-1`) Some(x)
                else None
              }

            case BitOr() if isIntegral(typ) =>
              val `-1` = MinusOne
              if (x == `-1` || y == `-1`) Some(`-1`)
              else {
                val `0` = Zero
                if (x == `0`) Some(y)
                else if (y == `0`) Some(x)
                else None
              }

            case BitXOr() if isIntegral(typ) =>
              val `0` = Zero
              if (x == y) Some(`0`)
              else if (x == `0`) Some(y)
              else if (y == `0`) Some(x)
              else None

            case _ =>
              None
          }

        case _ =>
          None
      }
    }

  private[this] def applyRules(ir: IR): Option[IR] =
    ir match {
      case ApplyIR(name, _, args, _, errorID) =>
        name match {
          case "indexArray" =>
            args match {
              case Seq(a, i @ I32(v)) if v >= 0 => Some(ArrayRef(a, i, errorID))
              case _ => None
            }

          case "contains" =>
            args match {
              case Seq(CastToArray(x), element) if x.typ.isInstanceOf[TSet] =>
                Some(invoke("contains", TBoolean, x, element))

              case Seq(Literal(t, v), element) if t.isInstanceOf[TArray] =>
                Some(
                  invoke(
                    "contains",
                    TBoolean,
                    Literal(
                      TSet(t.asInstanceOf[TArray].elementType),
                      v.asInstanceOf[IndexedSeq[_]].toSet,
                    ),
                    element,
                  )
                )

              case Seq(ToSet(x), element) if x.typ.isInstanceOf[TArray] =>
                Some(invoke("contains", TBoolean, x, element))

              case _ =>
                None
            }
          case "annotate" =>
            args match {
              case Seq(s, MakeStruct(fields)) => Some(InsertFields(s, fields))
              case _ => None
            }

          case _ => None
        }

      // `land` and `lor` implement Kleene logic: an operand of False (resp. True) determines
      // the result even when the other operand is missing. Both evaluate left-to-right,
      // however, so a discarded left operand must still be evaluated for its side effects
      // (such as Die); the right operand is never evaluated when the left short-circuits.
      case ApplySpecial("land", _, Seq(l, r), _, _) =>
        (l, r) match {
          case (_: True, _) => Some(r)
          case (_: False, _) => Some(l)
          case (_: NA, _: False) => Some(r)
          case (_, _: True) => Some(l)
          case _ => if (l == r) Some(l) else None
        }

      case ApplySpecial("lor", _, Seq(l, r), _, _) =>
        (l, r) match {
          case (_: False, _) => Some(r)
          case (_: True, _) => Some(l)
          case (_: NA, _: True) => Some(r)
          case (_, _: False) => Some(l)
          case _ => if (l == r) Some(l) else None
        }

      case _: ApplyBinaryPrimOp =>
        binaryPrimOpRules(ir)

      case ApplyUnaryPrimOp(f, x) =>
        (f, x) match {
          case (Bang, ApplyComparisonOp(op: ComparisonOp[Boolean @unchecked], l, r)) =>
            Some(ApplyComparisonOp(ComparisonOp.negate(op), l, r))

          case (_, If(pred, cnsq, altr)) =>
            Some(If(pred, ApplyUnaryPrimOp(f, cnsq), ApplyUnaryPrimOp(f, altr)))

          case (Negate, I32(y)) => Some(I32(-y))
          case (Negate, I64(y)) => Some(I64(-y))
          case (Negate, F32(y)) => Some(F32(-y))
          case (Negate, F64(y)) => Some(F64(-y))

          // these ops are their own inverse
          case (Negate | BitNot | Bang, ApplyUnaryPrimOp(g, y)) if g == f =>
            Some(y)

          case (_, _: NA) =>
            Some(NA(ir.typ))

          case _ =>
            None
        }

      case ApplyComparisonOp(op, l, r) =>
        val structural = op match {
          case EQ =>
            (l, r) match {
              case (_, _: True) => Some(l)
              case (_: True, _) => Some(r)
              case (_, _: False) => Some(ApplyUnaryPrimOp(Bang, l))
              case (_: False, _) => Some(ApplyUnaryPrimOp(Bang, r))
              case _ => None
            }
          case _ =>
            None
        }

        structural.orElse {
          def isFloating(t: Type) = t == TFloat32 || t == TFloat64
          if ((!isFloating(l.typ) && l.typ == r.typ) && l == r)
            op match {
              case LT => Some(False())
              case LTEQ => Some(True())
              case EQ => Some(True())
              case GTEQ => Some(True())
              case GT => Some(False())
              case NEQ => Some(False())
              case _ => None
            }
          else None
        }

      case BlockMatrixToValueApply(
            ValueToBlockMatrix(child, IndexedSeq(_, ncols), _),
            functions.GetElement(Seq(i, j)),
          ) => child.typ match {
          case TArray(_) => Some(ArrayRef(child, I32((i * ncols + j).toInt)))
          case TNDArray(_, _) => Some(NDArrayRef(child, IndexedSeq(i, j), ErrorIDs.NO_ERROR))
          case TFloat64 => Some(child)
          case _ => None
        }

      case _ =>
        None
    }

  // does `If(x, _, _)` reduce to one of its children (or NA) by the rules below?
  private[this] def collapses(x: IR): Boolean =
    x match {
      case True() | False() | _: NA => true
      case _ => false
    }

  private[this] def conditionalRules(ir: IR): Option[IR] =
    ir match {
      case Coalesce(values) =>
        if (values.length == 1) Some(values.head)
        else if (values.exists(_.isInstanceOf[NA])) {
          // an NA can never supply the value
          val defined = values.filterNot(_.isInstanceOf[NA])
          if (defined.isEmpty) Some(values.head)
          else Some(Coalesce(defined))
        } else if (values.exists(_.isInstanceOf[Coalesce])) {
          val idx = values.indexWhere(_.isInstanceOf[Coalesce])

          Some(Coalesce(
            ArraySeq.newBuilder[IR]
              .++=(values.view.take(idx))
              .++=(values(idx).asInstanceOf[Coalesce].values)
              .++=(values.view.drop(idx + 1))
              .result()
          ))
        } else {
          val idx = values.indexWhere(isDefinitelyDefined)
          if (idx >= 0 && idx < values.length - 1) Some(Coalesce(values.take(idx + 1)))
          else None
        }

      case IsNA(a) =>
        a match {
          case _: NA => Some(True())

          case ApplyBinaryPrimOp(_, l, r) => Some(IsNA(l) || IsNA(r))
          case ApplyComparisonOp(op, l, r) if op.strict => Some(IsNA(l) || IsNA(r))
          case ApplyUnaryPrimOp(_, a) => Some(IsNA(a))

          // `IsNA` observes only missingness: nodes that are missing iff a single strict child
          // is missing are transparent to it, so test the child directly.
          case SelectFields(struct, _) => Some(IsNA(struct))
          case InsertFields(struct, _, _) => Some(IsNA(struct))
          case Cast(x, _) => Some(IsNA(x))
          case CastRename(x, _) => Some(IsNA(x))
          case CastToArray(x) => Some(IsNA(x))
          case ToArray(s) => Some(IsNA(s))
          case ToStream(x, _) => Some(IsNA(x))
          case ToSet(s) => Some(IsNA(s))
          case ToDict(s) => Some(IsNA(s))
          case ArraySort(s, _, _, _) => Some(IsNA(s))
          case StreamMap(s, _, _) => Some(IsNA(s))
          case StreamFilter(s, _, _) => Some(IsNA(s))
          case StreamFlatMap(s, _, _) => Some(IsNA(s))

          // nodes with more than one strict child are transparent when every other child is
          // definitely defined
          case StreamTake(s, n) if isDefinitelyDefined(n) => Some(IsNA(s))
          case StreamDrop(s, n) if isDefinitelyDefined(n) => Some(IsNA(s))
          case ArraySlice(arr, start, stop, step, _)
              if isDefinitelyDefined(start) && stop.forall(isDefinitelyDefined)
                && isDefinitelyDefined(step) =>
            Some(IsNA(arr))

          // bindings are unaffected; move `IsNA` closer to the source of missingness
          case Block(bindings, body) => Some(Block(bindings, IsNA(body)))
          case If(predicate, cnsq, altr) =>
            Some(Coalesce(FastSeq(If(predicate, IsNA(cnsq), IsNA(altr)), True())))

          case _ =>
            if (isDefinitelyDefined(a)) Some(False())
            else None
        }

      case If(pred, cnsq, altr) =>
        (pred, cnsq, altr) match {
          case (_: NA, _, _) =>
            Some(NA(ir.typ))

          case (True(), _, _) =>
            Some(cnsq)

          case (False(), _, _) =>
            Some(altr)

          case (ApplyUnaryPrimOp(Bang, c), _, _) =>
            Some(If(c, altr, cnsq))

          case _ if cnsq == altr && cnsq.typ != TVoid =>
            if (isDefinitelyDefined(pred)) Some(cnsq)
            else Some(If(IsNA(pred), NA(cnsq.typ), cnsq))

          case (_, True(), False()) =>
            Some(pred)

          case (_, False(), True()) =>
            Some(ApplyUnaryPrimOp(Bang, pred))

          case (_, If(c2, cnsq2, _), _) if pred == c2 =>
            Some(If(pred, cnsq2, altr))

          case (_, _, If(c2, _, altr2)) if pred == c2 =>
            Some(If(pred, cnsq, altr2))

          case (IsNA(a), _: NA, b) if a == b =>
            Some(b)

          // case-of-case, last: it duplicates `cnsq` and `altr`, so it must not
          // preempt the shrinking rules above and only fires when a duplicate
          // provably collapses or both branches are atoms
          case (If(p, c, a), _, _)
              if (collapses(c) && collapses(a))
                || (cnsq.isInstanceOf[Atom] && altr.isInstanceOf[Atom]) =>
            Some(If(p, If(c, cnsq, altr), If(a, cnsq.deepCopy, altr.deepCopy)))

          case _ =>
            None
        }

      case Switch(x, default, cases) =>
        x match {
          case I32(v) => Some(if (v >= 0 && v < cases.length) cases(v) else default)
          case _ if cases.isEmpty && isDefinitelyDefined(x) => Some(default)
          case _ => None
        }

      case _ =>
        None
    }

  private[this] def arrayRules(ir: IR): Option[IR] =
    ir match {
      case ArrayLen(child) =>
        child match {
          case MakeArray(args, _) => Some(I32(args.length))
          case ToArray(s) => Some(StreamLen(s))
          case StreamFlatMap(a, _, MakeArray(args, _)) => Some(I32(args.length) * ArrayLen(a))
          case ArraySort(a, _, _, _) => Some(ArrayLen(ToArray(a)))
          case ToArray(MakeStream(args, _, _)) => Some(I32(args.length))
          case GetField(TableCollect(tc), "rows") => Some(Cast(TableCount(tc), TInt32))
          case If(pred, cnsq, altr) => Some(If(pred, ArrayLen(cnsq), ArrayLen(altr)))
          case _ => None
        }

      // Non-negative constant unit-step slices are windows and need not copy:
      // a[i::1] == drop(a, i) and a[i:j:1] == take(drop(a, i), max(0, j - i)); python
      // slice semantics clamp both bounds to the array length, as do take and drop.
      case ArraySlice(a, I32(i), stop, I32(1), _) if i >= 0 =>
        (a, stop) match {
          case (ToArray(s), Some(I32(j))) if j >= 0 =>
            Some(ToArray(StreamTake(StreamDrop(s, I32(i)), I32((j - i).max(0)))))

          case (ToArray(s), None) =>
            Some(ToArray(StreamDrop(s, I32(i))))

          case (_, Some(ArrayLen(`a`)) | None) =>
            if (i == 0) Some(a)
            else if (i == 1) Some(ToArray(StreamDrop(ToStream(a), I32(1)))) // tail
            else None

          case _ =>
            None
        }

      case ArrayRef(MakeArray(args, _), I32(i), _) if i >= 0 && i < args.length =>
        Some(args(i))

      case _ =>
        None
    }

  private[this] def streamRules(ir: IR): Option[IR] =
    ir match {
      case StreamAgg(_, _, query) if !ContainsAgg(query) =>
        Some(query)

      case StreamAggScan(a, name, query) if !ContainsScan(query) =>
        Some(StreamMap(a, name, query))

      case StreamFlatMap(stream, x, f) =>
        stream match {
          case Block(bindings, body) =>
            Some(Block(bindings, StreamFlatMap(body, x, f)))

          case StreamMap(a, y, g) =>
            Some(StreamFlatMap(a, y, Let(FastSeq(x -> g), f)))

          case StreamFlatMap(a, y, g) =>
            Some(StreamFlatMap(a, y, StreamFlatMap(g, x, f)))

          case _: NA =>
            Some(NA(ir.typ))

          case _ => None
        }

      // FIXME: Unqualify when StreamFold supports folding over stream of streams
      case StreamFold(s, zero, accum, elem, f) =>
        s match {
          case Block(bindings, body) =>
            Some(Block(bindings, StreamFold(body, zero, accum, elem, f)))

          case StreamFilter(a, name, cond) if TIterable.elementType(a.typ).isRealizable =>
            val ref = Ref(name, TIterable.elementType(a.typ))
            val body = If(cond, Subst(f, BindingEnv.eval(elem -> ref)), Ref(accum, zero.typ))
            Some(StreamFold(a, zero, accum, name, body))

          case StreamFlatMap(a, name, g) if TIterable.elementType(a.typ).isRealizable =>
            val accum2 = Ref(freshName(), zero.typ)
            val inner = StreamFold(g, accum2, accum, elem, f)
            Some(StreamFold(a, zero, accum2.name, name, inner))

          case StreamMap(a, name, g) if TIterable.elementType(a.typ).isRealizable =>
            Some(StreamFold(a, zero, accum, name, Let(FastSeq(elem -> g), f)))

          case _: NA =>
            Some(NA(ir.typ))

          case _ =>
            None
        }

      case StreamFor(stream, x, f) =>
        stream match {
          case Block(bindings, value) =>
            Some(Block(bindings, StreamFor(value, x, f)))

          case StreamMap(inner, y, g) =>
            Some(StreamFor(inner, y, Let(FastSeq(x -> g), f)))

          case StreamFilter(inner, y, p) =>
            val elemType = TIterable.elementType(inner.typ)
            val body = If(p, Subst(f, BindingEnv.eval(x -> Ref(y, elemType))), Void())
            Some(StreamFor(inner, y, body))

          case StreamFlatMap(inner, y, g) =>
            Some(StreamFor(inner, y, StreamFor(g, x, f)))

          case _ =>
            f match {
              case _: Void => Some(Void())
              case _ => None
            }
        }

      case StreamFilter(stream, x, cond) =>
        stream match {
          case Block(bindings, body) =>
            Some(Block(bindings, StreamFilter(body, x, cond)))

          case ArraySort(a, left, right, lessThan) =>
            Some(ArraySort(StreamFilter(a, x, cond), left, right, lessThan))

          case ToStream(ArraySort(a, left, right, lessThan), memoryManaged) =>
            Some(
              ToStream(
                ArraySort(StreamFilter(a, x, cond), left, right, lessThan),
                memoryManaged,
              )
            )

          case StreamFlatMap(child, flatMapName, flatMapBody) =>
            Some(StreamFlatMap(child, flatMapName, StreamFilter(flatMapBody, x, cond)))

          // filtering preserves the stream type, so the NA child needs no retyping
          case _: NA =>
            Some(stream)

          case _ =>
            cond match {
              case _: True => Some(stream)
              // an empty take preserves the missingness of `stream`
              case _: False => Some(StreamTake(stream, I32(0)))
              case _ => None
            }
        }

      case StreamLen(stream) =>
        stream match {
          case MakeStream(args, _, _) =>
            Some(I32(args.length))

          case Block(bindings, body) =>
            Some(Block(bindings, StreamLen(body)))

          case StreamMap(s, _, _) =>
            Some(StreamLen(s))

          case StreamFlatMap(a, name, body) =>
            Some(streamSumIR(StreamMap(a, name, StreamLen(body))))

          case StreamGrouped(a, groupSize) =>
            Some(groupSize.bind(groupSizeRef =>
              (StreamLen(a) + groupSizeRef - 1) floorDiv groupSizeRef
            ))

          case StreamRange(start, stop, I32(step), _, _) if step != 0 =>
            val ceilDiv = -((start.toL - stop.toL) floorDiv step.toLong)
            Some(bindIR(ceilDiv.toI)(x => x.if_(x > 0)(0)))

          case _ =>
            None
        }

      case StreamMap(stream, x, f) =>
        stream match {
          case Block(bindings, body) =>
            Some(Block(bindings, StreamMap(body, x, f)))

          case MakeStream(Seq(), _, _) =>
            Some(MakeStream(FastSeq(), tcoerce[TStream](ir.typ)))

          case StreamMap(s, y, g) =>
            Some(StreamMap(s, y, Let(FastSeq(x -> g), f)))

          case StreamZip(as, names, zipBody, b, errorID) =>
            Some(StreamZip(as, names, Let(FastSeq(x -> zipBody), f), b, errorID))

          case StreamFlatMap(child, flatMapName, flatMapBody) =>
            Some(StreamFlatMap(child, flatMapName, StreamMap(flatMapBody, x, f)))

          case r: RunAggScan =>
            Some(r.copy(result = Let(FastSeq(x -> r.result), f)))

          case _: NA =>
            Some(NA(ir.typ))

          case _ =>
            f match {
              case Ref(`x`, _) => Some(stream)
              case _ => None
            }
        }

      case StreamZip(as, names, body, _, _) if as.length == 1 =>
        Some(StreamMap(as.head, names.head, body))

      // provably-empty constant ranges
      case StreamRange(I32(start), I32(stop), I32(step), _, _)
          if (step > 0 && start >= stop) || (step < 0 && start <= stop) =>
        Some(MakeStream(FastSeq(), TStream(TInt32)))

      case _ =>
        None
    }

  private[this] def streamSubsetRules(ir: IR): Option[IR] =
    ir match {
      case StreamTakeWhile(s, _, cond) =>
        cond match {
          case True() => Some(s)
          // take nothing, preserving the missingness of `s`
          case False() => Some(StreamTake(s, I32(0)))
          case _ => None
        }

      case StreamDropWhile(s, _, cond) =>
        cond match {
          // drop everything, preserving the missingness of `s`
          case True() => Some(StreamTake(s, I32(0)))
          case False() => Some(s)
          case _ => None
        }

      case StreamTake(s, n) =>
        s match {
          case ToStream(a, _) if (n match {
                case ArrayLen(`a`) => true
                case _ => false
              }) =>
            Some(s)

          // take(range(start, stop, step), k) is the range that stops after k steps, ie. at
          // start + k * step, clamped to the original stop. The clamped stop always lies between
          // start and stop, so it fits in an Int; compute in Long to avoid overflow. Rewriting
          // requires the range be valid, ie. step != 0, and the take be valid, ie. k >= 0.
          case StreamRange(I32(start), I32(stop), I32(step), memManaged, errorID) =>
            n match {
              case I32(k) if k >= 0 && step != 0 =>
                val taken = start.toLong + k.toLong * step
                val newStop = if (step > 0) taken.min(stop.toLong) else taken.max(stop.toLong)
                Some(StreamRange(I32(start), I32(newStop.toInt), I32(step), memManaged, errorID))
              case _ => None
            }

          case StreamMap(stream, name, body) =>
            Some(StreamMap(StreamTake(stream, n), name, body))

          case MakeStream(args, typ, memManaged) =>
            n match {
              case I32(k) if k >= 0 => Some(MakeStream(args.take(k), typ, memManaged))
              case _ => None
            }

          case StreamTake(inner, I32(m)) =>
            n match {
              case I32(k) if k >= 0 && m >= 0 =>
                Some(if (m <= k) s else StreamTake(inner, n))
              case _ => None
            }

          // element i of a zip depends only on element i of each input, for every zip behavior.
          // Bind the count so that it is evaluated once.
          case StreamZip(as, names, body, behavior, errorID) =>
            Some(n.bind(len =>
              StreamZip(as.map(StreamTake(_, len)), names, body, behavior, errorID)
            ))

          case _ =>
            n match {
              case I32(0) if isDefinitelyDefined(s) =>
                Some(MakeStream(FastSeq(), s.typ.asInstanceOf[TStream]))
              case _ =>
                None
            }
        }

      case StreamDrop(s, n) =>
        s match {
          // drop(to_stream(a), len(a)) is empty but must preserve the missingness of `a`
          case ToStream(a, _) if (n match {
                case ArrayLen(`a`) => true
                case _ => false
              }) =>
            Some(StreamTake(s, I32(0)))
          // drop(range(start, stop, step), k) is the range that starts after k steps, ie. at
          // start + k * step, clamped to the original stop. The clamped start always lies
          // between start and stop, so it fits in an Int; compute in Long to avoid overflow.
          // Rewriting requires the range be valid, ie. step != 0, and the drop be valid,
          // ie. k >= 0.
          case StreamRange(I32(start), I32(stop), I32(step), memManaged, errorID) =>
            n match {
              case I32(k) if k >= 0 && step != 0 =>
                val skipped = start.toLong + k.toLong * step
                val newStart = if (step > 0) skipped.min(stop.toLong) else skipped.max(stop.toLong)
                Some(StreamRange(I32(newStart.toInt), I32(stop), I32(step), memManaged, errorID))
              case _ => None
            }

          case StreamMap(stream, name, body) =>
            Some(StreamMap(StreamDrop(stream, n), name, body))

          case MakeStream(args, typ, memManaged) =>
            n match {
              case I32(k) if k >= 0 => Some(MakeStream(args.drop(k), typ, memManaged))
              case _ => None
            }

          case StreamDrop(inner, I32(m)) =>
            n match {
              case I32(k) if k >= 0 && m >= 0 =>
                // saturating add: dropping more than the stream's length is legal
                Some(
                  if (k == 0) s
                  else StreamDrop(inner, I32((m.toLong + k).min(Int.MaxValue).toInt))
                )
              case _ => None
            }

          // canonicalise windows as take-of-drop so that takes and drops fuse
          case StreamTake(inner, I32(m)) =>
            n match {
              case I32(k) if k >= 0 && m >= 0 =>
                Some(
                  if (k == 0) s
                  else StreamTake(StreamDrop(inner, n), I32((m - k).max(0)))
                )
              case _ => None
            }

          case StreamZip(as, names, body, behavior, errorID) =>
            Some(n.bind(len =>
              StreamZip(as.map(StreamDrop(_, len)), names, body, behavior, errorID)
            ))

          case _ =>
            n match {
              case I32(0) => Some(s)
              case _ => None
            }
        }

      case _ =>
        None
    }

  private[this] def castRules(ir: IR): Option[IR] =
    ir match {
      case Cast(child, t) =>
        child match {
          case _ if child.typ == t => Some(child)
          case Cast(x, _) if x.typ == t && Casts.get(t, child.typ).isLossless => Some(x)
          case _ => None
        }

      case CastRename(child, t) =>
        child match {
          case _ if child.typ == t => Some(child)
          case CastRename(x, _) => Some(CastRename(x, t))
          case _ => None
        }

      case CastToArray(x) if x.typ.isInstanceOf[TArray] =>
        Some(x)

      case ToArray(child) =>
        child match {
          case ToStream(a, _) if a.typ.isInstanceOf[TArray] =>
            Some(a)

          case ToStream(a, _) if a.typ.isInstanceOf[TSet] || a.typ.isInstanceOf[TDict] =>
            Some(CastToArray(a))

          case MakeStream(args, _, _) =>
            Some(MakeArray(args, tcoerce[TArray](ir.typ)))

          case If(pred, cnsq, altr) =>
            Some(If(pred, ToArray(cnsq), ToArray(altr)))

          case _: NA =>
            Some(NA(ir.typ))

          case _ =>
            None
        }

      case ToStream(child, memManaged) if !memManaged =>
        child match {
          case ToArray(s) if s.typ.isInstanceOf[TStream] =>
            Some(s)

          case Block(bindings, ToArray(x)) if x.typ.isInstanceOf[TStream] =>
            Some(Block(bindings, x))

          case MakeArray(args, t) =>
            Some(MakeStream(args, TStream(t.elementType)))

          case _: NA =>
            Some(NA(ir.typ))

          case _ =>
            None
        }

      case _ =>
        None
    }

  private[this] def ndarrayRules(ir: IR): Option[IR] =
    ir match {
      case MakeNDArray(data, shape, rowMajor, errorId) =>
        data match {
          case ToArray(someStream) =>
            Some(MakeNDArray(someStream, shape, rowMajor, errorId))

          case ToStream(someArray, _) =>
            Some(MakeNDArray(someArray, shape, rowMajor, errorId))

          case _ =>
            None
        }

      case NDArrayShape(child) =>
        child match {
          case MakeNDArray(data, shape, _, _) =>
            Some(If(IsNA(data), NA(shape.typ), shape))

          case NDArrayMap(nd, _, _) =>
            Some(NDArrayShape(nd))

          case _ =>
            None
        }

      case NDArrayMap(child, elem, body) =>
        body match {
          case Ref(`elem`, _) =>
            Some(child)

          case _ =>
            child match {
              case NDArrayMap(inner, innerName, innerBody) =>
                Some(NDArrayMap(inner, innerName, Let(FastSeq(elem -> innerBody), body)))

              case _ =>
                None
            }
        }

      case _ =>
        None
    }

  private[this] def structRules(ir: IR): Option[IR] =
    ir match {
      case GetField(child, name) =>
        child match {
          case MakeStruct(fields) =>
            val (_, x) = fields.find { case (n, _) => n == name }.get
            Some(x)

          case InsertFields(old, fields, _) =>
            fields.find { case (n, _) => n == name } match {
              case Some((_, x)) => Some(x)
              case None => Some(GetField(old, name))
            }

          case SelectFields(old, _) =>
            Some(GetField(old, name))

          case TableCollect(tc) if name == "global" =>
            Some(TableGetGlobals(tc))

          case _ => None
        }

      case outer @ InsertFields(child, fields, fieldOrder) =>
        child match {
          case InsertFields(base, fields1, fieldOrder1) =>
            val fieldsSet = fields.map(_._1).toSet
            val newFields =
              fields1.filter { case (name, _) => !fieldsSet.contains(name) } ++ fields
            (fieldOrder1, fieldOrder) match {
              case (Some(fo1), None) =>
                val fields1Set = fo1.toSet
                val fo = fo1 ++ fields.map(_._1).filter(!fields1Set.contains(_))
                Some(InsertFields(base, newFields, Some(fo)))
              case (_, Some(_)) =>
                Some(InsertFields(base, newFields, fieldOrder))
              case _ =>
                val resultFieldOrder = outer.typ.fieldNames
                Some(InsertFields(base, newFields, Some(resultFieldOrder)))
            }

          case MakeStruct(fields1) =>
            val fields1Map = fields1.toMap
            val fields2Map = fields.toMap
            fieldOrder match {
              case Some(fo) =>
                Some(MakeStruct(fo.map(f => f -> fields2Map.getOrElse(f, fields1Map(f)))))
              case None =>
                val finalFields = fields1.map { case (name, fieldIR) =>
                  name -> fields2Map.getOrElse(name, fieldIR)
                } ++
                  fields.filter { case (name, _) => !fields1Map.contains(name) }
                Some(MakeStruct(finalFields))
            }

          case SelectFields(struct, selectFields) =>
            if (fields.isEmpty)
              fieldOrder match {
                case Some(fo) => Some(SelectFields(struct, fo))
                case None => Some(child)
              }
            else if (fields.exists { case (name, f) => f == GetField(struct, name) }) {
              val resultFields = outer.typ.fieldNames
              val insertNames = fields.map(_._1).toSet
              val (oldFields, newFields) =
                fields.partition { case (name, f) => f == GetField(struct, name) }
              val preservedFields =
                selectFields.filter(f => !insertNames.contains(f)) ++ oldFields.map(_._1)
              Some(InsertFields(
                SelectFields(struct, preservedFields),
                newFields,
                Some(resultFields),
              ))
            } else None

          case _ =>
            if (fields.isEmpty && fieldOrder.isEmpty) Some(child)
            else if (
              outer.typ == child.typ &&
              fields.forall { case (name, value) => value == GetField(child, name) }
            )
              // re-inserting existing fields with their own values is a no-op
              Some(child)
            else None
        }

      case _ =>
        None
    }

  private[this] def blockRules(ir: IR): Option[IR] =
    ir match {
      case Block(bindings, body) =>
        body match {
          case Block(ys, innerBody) => Some(Block(bindings ++ ys, innerBody))
          case _ => bindings match {
              case Seq() =>
                Some(body)
              // assumes `NormalizeNames` has been run before this.
              case Block.Nested(i, bindings) =>
                def numBindings(b: Binding): Int =
                  b.value match {
                    case let: Block => 1 + let.bindings.length
                    case _ => 1
                  }

                val newBindings = ArraySeq.newBuilder[Binding]
                newBindings.sizeHint(bindings.view.map(numBindings).sum)

                newBindings ++= bindings.view.take(i)

                bindings.view.drop(i).foreach {
                  case Binding(name, ir: Block, scope) =>
                    if (scope == EVAL) newBindings ++= ir.bindings
                    else ir.bindings.foreach { case Binding(name, value, s) =>
                      if (s != EVAL) fatal(s"Simplify: found nested $s binding in $scope")
                      newBindings += Binding(name, value, scope)
                    }
                    newBindings += Binding(name, ir.body, scope)
                  case binding =>
                    newBindings += binding
                }

                Some(Block(newBindings.result(), body))

              case Block.Insert(
                    before,
                    Binding(name, x @ InsertFields(old, newFields, _), Scope.EVAL),
                    after,
                  )
                  if x.typ.size < 500 && {
                    val r = Ref(name, x.typ)
                    val nfSet = newFields.map(_._1).toSet

                    def allRefsCanBePassedThrough(ir1: IR): Boolean = ir1 match {
                      case GetField(`r`, _) => true
                      case InsertFields(`r`, inserted, _) =>
                        inserted.forall { case (_, toInsert) =>
                          allRefsCanBePassedThrough(toInsert)
                        }
                      case SelectFields(`r`, fds) => fds.forall(f => !nfSet.contains(f))
                      case `r` => false
                      case _: TableAggregate => true
                      case _: MatrixAggregate => true
                      case _ => ir1.children
                          .zipWithIndex
                          .forall {
                            case (child: IR, idx) =>
                              Binds(ir1, name, idx) || allRefsCanBePassedThrough(child)
                            case _ => true
                          }
                    }

                    allRefsCanBePassedThrough(Block(after, body))
                  } =>
                val fieldNames = newFields.map(_._1)
                val newFieldMap = newFields.toMap
                val newFieldRefs = newFieldMap.map { case (k, ir) =>
                  (k, Ref(freshName(), ir.typ))
                } // cannot be mapValues, or genUID() gets run for every usage!

                def copiedNewFieldRefs(): IndexedSeq[(String, IR)] =
                  fieldNames.map(name => (name, newFieldRefs(name)))

                def rewrite(ir1: IR): IR = ir1 match {
                  case GetField(Ref(`name`, _), fd) => newFieldRefs.get(fd) match {
                      case Some(r) => r.ir
                      case None => GetField(Ref(name, old.typ), fd)
                    }
                  case ins @ InsertFields(Ref(`name`, _), fields, _) =>
                    val newFieldSet = fields.map(_._1).toSet
                    InsertFields(
                      Ref(name, old.typ),
                      copiedNewFieldRefs().filter { case (name, _) =>
                        !newFieldSet.contains(name)
                      }
                        ++ fields.map { case (name, ir) => (name, rewrite(ir)) },
                      Some(ins.typ.fieldNames),
                    )

                  case SelectFields(Ref(`name`, _), fds) =>
                    SelectFields(
                      InsertFields(
                        Ref(name, old.typ),
                        copiedNewFieldRefs(),
                        Some(x.typ.fieldNames),
                      ),
                      fds,
                    )
                  case ta: TableAggregate => ta
                  case ma: MatrixAggregate => ma
                  case _ => ir1.mapChildrenWithIndex {
                      case (child: IR, idx) =>
                        if (Binds(ir1, name, idx)) child else rewrite(child)
                      case (child, _) => child
                    }
                }

                val bs = ArraySeq.newBuilder[Binding]
                bs ++= before
                fieldNames.foreach(f => bs += Binding(newFieldRefs(f).name, newFieldMap(f)))
                bs += Binding(name, old)
                Some(
                  Block(bs.result(), rewrite(Block(after, body)))
                )

              case _ => None
            }
        }

      case _ =>
        None
    }

  private[this] def selectRules(ir: IR): Option[IR] =
    ir match {
      case SelectFields(old, fields) =>
        old match {
          case _ if tcoerce[TStruct](old.typ).fieldNames == fields =>
            Some(old)

          case SelectFields(inner, _) =>
            Some(SelectFields(inner, fields))

          case MakeStruct(structFields) =>
            val makeStructFields = structFields.toMap
            Some(MakeStruct(fields.map(f => f -> makeStructFields(f))))

          case InsertFields(struct, insertFields, _) =>
            val selectSet = fields.toSet
            val insertFields2 =
              insertFields.filter { case (fName, _) => selectSet.contains(fName) }
            val structSet = struct.typ.asInstanceOf[TStruct].fieldNames.toSet
            val selectFields2 = fields.filter(structSet.contains)
            val x2 = InsertFields(
              SelectFields(struct, selectFields2),
              insertFields2,
              Some(fields),
            )
            assert(x2.typ == ir.typ)
            Some(x2)

          case _ =>
            if (fields.isEmpty && isDefinitelyDefined(old)) Some(makestruct())
            else None
        }

      case MakeStructOfGetField(o, newNames) =>
        val select = SelectFields(o, newNames.map(_._1))
        Some(CastRename(select, select.typ.rename(newNames.toMap)))

      case GetTupleElement(MakeTuple(xs), idx) =>
        Some(xs.find(_._1 == idx).get._2)

      case _ =>
        None
    }

  private[this] def countRules(ir: IR): Option[IR] =
    ir match {
      case TableCount(child) =>
        child match {
          case MatrixColsTable(ColumnCount(nCols)) =>
            Some(I64(nCols.toLong))

          case PartitionCounts(counts) =>
            Some(I64(counts.sum))

          case CastMatrixToTable(c, _, _) =>
            Some(TableCount(MatrixRowsTable(c)))

          case TableMapGlobals(c, _) =>
            Some(TableCount(c))

          case TableMapRows(c, _) =>
            Some(TableCount(c))

          case TableRepartition(c, _, _) =>
            Some(TableCount(c))

          case TableUnion(children) =>
            Some(children.map(TableCount(_): IR).treeReduce(ApplyBinaryPrimOp(Add(), _, _)))

          case TableKeyBy(c, _, _, _) =>
            Some(TableCount(c))

          case TableOrderBy(c, _) =>
            Some(TableCount(c))

          case TableLeftJoinRightDistinct(c, _, _) =>
            Some(TableCount(c))

          case TableIntervalJoin(c, _, _, _) =>
            Some(TableCount(c))

          case TableRange(n, _) =>
            Some(I64(n.toLong))

          case TableParallelize(rowsAndGlobal, _) =>
            Some(Cast(ArrayLen(GetField(rowsAndGlobal, "rows")), TInt64))

          case TableRename(c, _, _) =>
            Some(TableCount(c))

          case TableAggregateByKey(c, _) =>
            Some(TableCount(TableDistinct(c)))

          case TableExplode(c, path) =>
            Some(
              TableAggregate(
                c,
                ApplyAggOp(
                  FastSeq(),
                  FastSeq(ArrayLen(CastToArray(path.foldLeft[IR](Ref(
                    TableIR.rowName,
                    c.typ.rowType,
                  )) {
                    case (comb, s) => GetField(comb, s)
                  })).toL),
                  Sum(),
                ),
              )
            )

          case TableRead(_, false, r: MatrixBGENReader) if r.params.includedVariants.isEmpty =>
            Some(I64(r.nVariants))

          case _ =>
            None
        }

      case MatrixCount(child) =>
        child match {
          case PartitionCountsOrColumnCount((maybeParts, maybeCols)) =>
            val rowCount = maybeParts match {
              case Some(pc) => I64(pc.sum)
              case None => TableCount(MatrixRowsTable(child))
            }
            val colCount = maybeCols match {
              case Some(cc) => I32(cc)
              case None => TableCount(MatrixColsTable(child)).toI
            }
            Some(MakeTuple.ordered(FastSeq(rowCount, colCount)))
          case MatrixMapRows(c, _) => Some(MatrixCount(c))
          case MatrixMapCols(c, _, _) => Some(MatrixCount(c))
          case MatrixMapEntries(c, _) => Some(MatrixCount(c))
          case MatrixFilterEntries(c, _) => Some(MatrixCount(c))
          case MatrixAnnotateColsTable(c, _, _) => Some(MatrixCount(c))
          case MatrixAnnotateRowsTable(c, _, _, _) => Some(MatrixCount(c))
          case MatrixRepartition(c, _, _) => Some(MatrixCount(c))
          case MatrixRename(c, _, _, _, _) => Some(MatrixCount(c))
          case _ => None
        }

      case _ =>
        None
    }

  private[this] def tableValueRules(ir: IR): Option[IR] =
    ir match {
      // TableGetGlobals should simplify very aggressively
      case TableGetGlobals(child) =>
        child match {
          case _ if child.typ.globalType == TStruct.empty =>
            Some(MakeStruct(FastSeq()))

          case TableKeyBy(c, _, _, _) =>
            Some(TableGetGlobals(c))

          case TableFilter(c, _) =>
            Some(TableGetGlobals(c))

          case TableHead(c, _) =>
            Some(TableGetGlobals(c))

          case TableRepartition(c, _, _) =>
            Some(TableGetGlobals(c))

          case TableJoin(child1, child2, _, _) =>
            Some(
              bindIRs(TableGetGlobals(child1), TableGetGlobals(child2)) { case Seq(g1, g2) =>
                MakeStruct(
                  g1.typ.asInstanceOf[TStruct].fields.map(f => f.name -> GetField(g1, f.name)) ++
                    g2.typ.asInstanceOf[TStruct].fields.map(f => f.name -> GetField(g2, f.name))
                )
              }
            )

          case TableMultiWayZipJoin(children, _, globalName) =>
            Some(makestruct(
              globalName ->
                MakeArray(children.map(TableGetGlobals), TArray(children.head.typ.globalType))
            ))
          case TableLeftJoinRightDistinct(c, _, _) =>
            Some(TableGetGlobals(c))

          case TableMapRows(c, _) =>
            Some(TableGetGlobals(c))

          case TableMapGlobals(c, newGlobals) =>
            Some(bindIR(TableGetGlobals(c)) { ref =>
              Subst(newGlobals, BindingEnv.eval(TableIR.globalName -> ref))
            })

          case TableExplode(c, _) =>
            Some(TableGetGlobals(c))

          case TableUnion(children) =>
            Some(TableGetGlobals(children.head))

          case TableDistinct(c) =>
            Some(TableGetGlobals(c))

          case TableAggregateByKey(c, _) =>
            Some(TableGetGlobals(c))

          case TableKeyByAndAggregate(c, _, _, _, _) =>
            Some(TableGetGlobals(c))

          case TableOrderBy(c, _) =>
            Some(TableGetGlobals(c))

          case TableRename(c, _, globalMap) =>
            if (globalMap.isEmpty) Some(TableGetGlobals(c))
            else Some(bindIR(TableGetGlobals(c)) { ref =>
              MakeStruct(c.typ.globalType.fieldNames.map { f =>
                globalMap.getOrElse(f, f) -> GetField(ref, f)
              })
            })
          case _ => None
        }

      case TableCollect(child) =>
        child match {
          case TableParallelize(x, _) =>
            Some(x)

          case TableOrderBy(orderChild, sortFields)
              if sortFields.forall(_.sortOrder == Ascending)
                && !orderChild.typ.key.startsWith(sortFields.map(_.field)) =>
            Some(orderChild.keyBy(FastSeq()).collect.bind { rowsAndGlobal =>
              makestruct(
                "rows" -> rowsAndGlobal
                  .get("rows")
                  .stream
                  .streamMap(row => maketuple(row.select(sortFields.map(_.field)), row))
                  .sort(_.get(0) < _.get(0))
                  .stream
                  .streamMap(_.get(1))
                  .toArray,
                "global" -> rowsAndGlobal.get("global"),
              )
            })

          case _ =>
            None
        }

      case TableAggregate(child, query) =>
        child match {
          case _ if child.typ.key.nonEmpty && !ContainsNonCommutativeAgg(query) =>
            Some(TableAggregate(TableKeyBy(child, FastSeq(), false), query))

          case TableOrderBy(orderChild, _) if !ContainsNonCommutativeAgg(query) =>
            if (orderChild.typ.key.isEmpty) Some(TableAggregate(orderChild, query))
            else Some(TableAggregate(TableKeyBy(orderChild, FastSeq(), false), query))

          case TableMapRows(mapChild, newRow) if !ContainsScan(newRow) =>
            val uid = freshName()
            Some(TableAggregate(
              mapChild,
              AggLet(
                uid,
                newRow,
                Subst(
                  query,
                  BindingEnv(agg = Some(Env(TableIR.rowName -> Ref(uid, newRow.typ)))),
                ),
                isScan = false,
              ),
            ))

          case _ =>
            None
        }

      // NOTE: The below rule should be reintroduced when it is possible to put an ArrayAgg
      // inside a TableAggregate
      // case TableAggregate(TableParallelize(rowsAndGlobal, _), query) =>
      //   rowsAndGlobal match {
      // match because we currently don't optimize MakeStruct through Let, and this is a common
      // pattern
      //     case MakeStruct(Seq((_, rows), (_, global))) =>
      //       Let("global", global, ArrayAgg(rows, "row", query))
      //     case other =>
      //       val uid = genUID()
      //       Let(uid,
      //         rowsAndGlobal,
      //         Let("global",
      //           GetField(Ref(uid, rowsAndGlobal.typ), "global"),
      //           ArrayAgg(GetField(Ref(uid, rowsAndGlobal.typ), "rows"), "row", query)))
      //   }

      case _ =>
        None
    }

  private[this] def tableRules(ctx: ExecuteContext, tir: TableIR): Option[TableIR] =
    tir match {
      case _: TableRename |
          _: TableFilter |
          _: TableOrderBy |
          _: TableKeyBy =>
        tableKeyAndFilterRules(tir)

      case _: TableMapRows |
          _: TableMapGlobals |
          _: TableUnion =>
        tableMapRules(tir)

      case _: MatrixRowsTable |
          _: MatrixColsTable =>
        matrixTableRules(tir)

      case _: TableRepartition |
          _: TableHead |
          _: TableTail |
          _: TableDistinct =>
        tableSubsetRules(tir)

      case _: TableKeyByAndAggregate |
          _: TableAggregateByKey |
          _: TableParallelize =>
        tableAggRules(ctx, tir)

      case _: TableFilterIntervals =>
        tableFilterIntervalsRules(ctx, tir)

      case _ =>
        None
    }

  private[this] def tableKeyAndFilterRules(tir: TableIR): Option[TableIR] =
    tir match {
      // TODO: Write more rules like this to bubble 'TableRename' nodes towards the root.
      case t @ TableRename(child, m1, m2) =>
        if (m1.isTrivial && m2.isTrivial) Some(child)
        else
          child match {
            case TableKeyBy(inner, keys, isSorted, nPartitions) =>
              Some(TableKeyBy(
                TableRename(inner, m1, m2),
                keys.map(t.rowF),
                isSorted,
                nPartitions,
              ))
            case _ => None
          }

      case TableFilter(child, pred) =>
        pred match {
          case _: True => Some(child)

          case _ =>
            child match {
              case TableRead(typ, _, tr) =>
                pred match {
                  case _: False | _: NA =>
                    Some(TableRead(typ, dropRows = true, tr))

                  case _ => None
                }

              case TableFilter(t, p1) =>
                Some(TableFilter(
                  t,
                  ApplySpecial(
                    "land",
                    ArraySeq.empty,
                    ArraySeq(p1, pred),
                    TBoolean,
                    ErrorIDs.NO_ERROR,
                  ),
                ))

              case TableKeyBy(c, key, isSorted, nPartitions) =>
                Some(TableKeyBy(
                  TableFilter(c, pred),
                  key,
                  isSorted,
                  nPartitions,
                ))

              case TableRepartition(c, n, strategy) =>
                Some(TableRepartition(TableFilter(c, pred), n, strategy))

              case TableOrderBy(c, sortFields) =>
                Some(TableOrderBy(TableFilter(c, pred), sortFields))

              case TableParallelize(rowsAndGlobal, nPartitions) =>
                import TableIR.{rowName, globalName}

                def mkFilteredRowsAndGlobal(rows: IR, global: Atom): IR =
                  makestruct(
                    "rows" -> rows
                      .stream
                      .filter(r => Subst(pred, BindingEnv.eval(rowName -> r, globalName -> global)))
                      .toArray,
                    "global" -> global,
                  )

                val newRowsAndGlobal =
                  rowsAndGlobal match {
                    case MakeStruct(Seq(("rows", rows), ("global", globalVal))) =>
                      globalVal.bind(mkFilteredRowsAndGlobal(rows, _))
                    case _ =>
                      M.eval {
                        rowsAndGlobal.flatMap { struct =>
                          struct
                            .get("global")
                            .map(mkFilteredRowsAndGlobal(struct.get("rows"), _))
                        }
                      }
                  }

                Some(TableParallelize(newRowsAndGlobal, nPartitions))

              case _ => None
            }
        }

      case TableOrderBy(TableKeyBy(c, _, false, _), sortFields) =>
        Some(TableOrderBy(c, sortFields))

      case TableKeyBy(child, key, isSorted, nPartitions) =>
        child match {
          case _ if key == child.typ.key =>
            Some(child)

          case TableOrderBy(c, _) if !isSorted =>
            Some(TableKeyBy(c, key, false, nPartitions))

          case TableKeyBy(c, _, innerIsSorted, _) =>
            if (!isSorted) Some(TableKeyBy(c, key, false, nPartitions))
            else if (innerIsSorted) Some(TableKeyBy(c, key, true, nPartitions))
            else None

          case _ => None
        }

      case _ =>
        None
    }

  private[this] def tableMapRules(tir: TableIR): Option[TableIR] =
    tir match {
      case TableMapRows(child, newRow) =>
        newRow match {
          case Ref(n, _) if n == TableIR.rowName =>
            Some(child)

          case MakeStruct(fields)
              if fields.length == child.typ.rowType.size
                && fields.zip(child.typ.rowType.fields).forall {
                  case ((_, ir), field) => ir == child.row.get(field.name)
                } =>
            val renamedPairs =
              for {
                (oldName, (newName, _)) <- child.typ.rowType.fieldNames zip fields
                if oldName != newName
              } yield oldName -> newName

            Some(TableRename(child, Map(renamedPairs: _*), Map.empty))

          case _ =>
            child match {
              case TableMapRows(inner, f) if !ContainsScan(newRow) =>
                val nr = f.bind(r => Subst(newRow, BindingEnv.eval(TableIR.rowName -> r)))
                Some(TableMapRows(inner, nr))
              case _ => None
            }
        }

      case TableMapGlobals(child, newGlobals) =>
        newGlobals match {
          case Ref(n, _) if n == TableIR.globalName =>
            Some(child)

          case _ =>
            child match {
              case TableMapGlobals(inner, ng1) =>
                Some(TableMapGlobals(
                  inner,
                  bindIR(ng1)(uid =>
                    Subst(newGlobals, BindingEnv.eval(TableIR.globalName -> uid))
                  ),
                ))
              case _ => None
            }
        }

      // flatten unions
      case TableUnion(children) =>
        if (children.exists(_.isInstanceOf[TableUnion]))
          Some(TableUnion(children.flatMap {
            case u: TableUnion => u.childrenSeq
            case c => Some(c)
          }))
        else None

      case _ =>
        None
    }

  private[this] def matrixTableRules(tir: TableIR): Option[TableIR] =
    tir match {
      case MatrixRowsTable(child) =>
        child match {
          case MatrixUnionRows(children) =>
            Some(TableUnion(children.map(MatrixRowsTable)))

          // Ignore column or row data that is immediately dropped
          case MatrixRead(typ, false, dropRows, reader) =>
            Some(MatrixRowsTable(MatrixRead(typ, dropCols = true, dropRows, reader)))

          case MatrixFilterRows(inner, pred) =>
            Some(MatrixRowsTable(inner).filter((_, row) =>
              Subst(pred, BindingEnv.eval(MatrixIR.rowName -> row))
            ))

          case MatrixMapGlobals(inner, newGlobals) =>
            Some(TableMapGlobals(MatrixRowsTable(inner), newGlobals))

          case MatrixMapCols(inner, _, _) =>
            Some(MatrixRowsTable(inner))

          case MatrixMapEntries(inner, _) =>
            Some(MatrixRowsTable(inner))

          case MatrixFilterEntries(inner, _) =>
            Some(MatrixRowsTable(inner))

          case MatrixFilterCols(inner, _) =>
            Some(MatrixRowsTable(inner))

          case MatrixAggregateColsByKey(inner, _, _) =>
            Some(MatrixRowsTable(inner))

          case MatrixChooseCols(inner, _) =>
            Some(MatrixRowsTable(inner))

          case MatrixCollectColsByKey(inner) =>
            Some(MatrixRowsTable(inner))

          case MatrixKeyRowsBy(inner, keys, isSorted) =>
            Some(TableKeyBy(MatrixRowsTable(inner), keys, isSorted))

          case _ => None
        }

      case MatrixColsTable(child) =>
        child match {
          case MatrixUnionRows(children) =>
            Some(MatrixColsTable(children(0)))

          // Ignore column or row data that is immediately dropped
          case MatrixRead(typ, dropCols, false, reader) =>
            Some(MatrixColsTable(
              MatrixRead(typ, dropCols, dropRows = true, reader)
            ))

          case MatrixMapCols(inner, newRow, newKey)
              if newKey.isEmpty
                && !ContainsAgg(newRow)
                && !ContainsScan(newRow) =>
            Some(MatrixColsTable(inner).mapRows((_, row) =>
              Subst(newRow, BindingEnv.eval(MatrixIR.colName -> row))
            ))

          case MatrixMapGlobals(inner, newGlobals) =>
            Some(TableMapGlobals(MatrixColsTable(inner), newGlobals))

          case MatrixMapRows(inner, _) =>
            Some(MatrixColsTable(inner))

          case MatrixMapEntries(inner, _) =>
            Some(MatrixColsTable(inner))

          case MatrixFilterEntries(inner, _) =>
            Some(MatrixColsTable(inner))

          case MatrixFilterRows(inner, _) =>
            Some(MatrixColsTable(inner))

          case MatrixAggregateRowsByKey(inner, _, _) =>
            Some(MatrixColsTable(inner))

          case MatrixKeyRowsBy(inner, _, _) =>
            Some(MatrixColsTable(inner))

          case _ => None
        }

      case _ =>
        None
    }

  private[this] def tableSubsetRules(tir: TableIR): Option[TableIR] =
    tir match {
      case TableRepartition(child, nParts, _) =>
        child match {
          case TableRange(nRows, _) => Some(TableRange(nRows, nParts))
          case _ => None
        }

      case TableHead(child, n) =>
        child match {
          case TableHead(c, m) =>
            Some(if (m <= n) child else TableHead(c, n))

          case MatrixColsTable(mc) if mc.typ.colKey.isEmpty =>
            Some(
              if (n > Int.MaxValue) child
              else MatrixColsTable(MatrixColsHead(mc, n.toInt))
            )

          case TableMapRows(c, newRow) =>
            Some(TableMapRows(TableHead(c, n), newRow))

          case TableRepartition(c, nPar, shuffle) =>
            Some(TableRepartition(TableHead(c, n), nPar, shuffle))

          case TableRange(nRows, nPar) =>
            Some(
              if (n < nRows) TableRange(n.toInt, (nPar.toFloat * n / nRows).toInt.max(1))
              else child
            )

          case TableMapGlobals(c, newGlobals) =>
            Some(TableMapGlobals(TableHead(c, n), newGlobals))

          case TableOrderBy(c, sortFields) // FIXME: https://github.com/hail-is/hail/issues/6234
              if !TableOrderBy.isAlreadyOrdered(sortFields, c.typ.key)
                && sortFields.forall(_.sortOrder == Ascending)
                && n < 256 =>
            // n < 256 is arbitrary for memory concerns
            Some(
              c
                .keyByAndAggregate(10, Some(1)) { (_, row) =>
                  val keyStruct = MakeStruct(sortFields.map(f => f.field -> row.get(f.field)))
                  makestruct("__row" -> ApplyAggOp(TakeBy(), n.toInt)(row, keyStruct))
                }((_, _) => makestruct())
                .explode("__row")
                .mapRows((_, row) => row.get("__row"))
            )

          case _ => None
        }

      case TableTail(child, n) =>
        child match {
          // both keep a suffix, so their composition keeps the smaller suffix
          case TableTail(c, m) =>
            Some(if (m <= n) child else TableTail(c, n))

          case MatrixColsTable(mc) if mc.typ.colKey.isEmpty =>
            Some(
              if (n > Int.MaxValue) child
              else MatrixColsTable(MatrixColsTail(mc, n.toInt))
            )

          // unlike `TableHead`, pushing a tail through a row map requires that the map contain
          // no scans: a scan's value on the kept rows depends on the dropped rows before them
          case TableMapRows(c, newRow) if !ContainsScan(newRow) =>
            Some(TableMapRows(TableTail(c, n), newRow))

          case TableRepartition(c, nPar, shuffle) =>
            Some(TableRepartition(TableTail(c, n), nPar, shuffle))

          case TableMapGlobals(c, newGlobals) =>
            Some(TableMapGlobals(TableTail(c, n), newGlobals))

          case _ => None
        }

      case TableDistinct(child) =>
        child match {
          // the rows of these children are already distinct by key
          case _: TableDistinct | _: TableAggregateByKey =>
            Some(child)

          case TableMapRows(inner, newRow) =>
            Some(TableMapRows(TableDistinct(inner), newRow))

          case TableLeftJoinRightDistinct(inner, right, root) =>
            Some(TableLeftJoinRightDistinct(
              TableDistinct(inner),
              right,
              root,
            ))

          case TableRepartition(inner, n, strategy) =>
            Some(TableRepartition(TableDistinct(inner), n, strategy))

          case _ => None
        }

      case _ =>
        None
    }

  private[this] def tableAggRules(ctx: ExecuteContext, tir: TableIR): Option[TableIR] =
    tir match {
      case TableKeyByAndAggregate(child, expr, newKey, _, _) =>
        (expr, newKey) match {
          case (MakeStruct(Seq()), k @ MakeStruct(_)) =>
            Some(
              TableDistinct(TableKeyBy(
                TableMapRows(TableKeyBy(child, FastSeq()), k),
                k.typ.fieldNames,
              ))
            )
          case _ =>
            if (
              (newKey == MakeStruct(child.typ.key.map(k => k -> child.row.get(k))) ||
                newKey == child.row.select(child.typ.key))
              && child.typ.key.nonEmpty
            )
              Some(TableAggregateByKey(child, expr))
            else None
        }

      case TableAggregateByKey(child, expr) =>
        child match {
          case x @ TableKeyBy(inner, keys, false, nPartitions) if !x.definitelyDoesNotShuffle =>
            Some(TableKeyByAndAggregate(
              inner,
              expr,
              MakeStruct(keys.map(k => k -> inner.row.get(k))),
              bufferSize = ctx.getFlag("grouped_aggregate_buffer_size").toInt,
              nPartitions = nPartitions,
            ))
          case _ => None
        }

      case TableParallelize(rowsAndGlobal, _) =>
        rowsAndGlobal match {
          case TableCollect(child) => Some(child)
          case _ => None
        }

      case _ =>
        None
    }

  private[this] def tableFilterIntervalsRules(ctx: ExecuteContext, tir: TableIR): Option[TableIR] =
    tir match {
      case TableFilterIntervals(child, intervals, keep) =>
        if (intervals.isEmpty) {
          if (keep) Some(TableFilter(child, False()))
          else Some(child)
        } else
          child match {
            // push down filter intervals nodes
            case TableFilter(c, pred) =>
              Some(TableFilter(
                TableFilterIntervals(c, intervals, keep),
                pred,
              ))

            case TableMapRows(c, newRow) if !ContainsScan(newRow) =>
              Some(TableMapRows(
                TableFilterIntervals(c, intervals, keep),
                newRow,
              ))

            case TableMapGlobals(c, newRow) =>
              Some(TableMapGlobals(
                TableFilterIntervals(c, intervals, keep),
                newRow,
              ))

            case TableRename(c, rowMap, globalMap) =>
              Some(TableRename(
                TableFilterIntervals(c, intervals, keep),
                rowMap,
                globalMap,
              ))

            case TableRepartition(c, n, strategy) =>
              Some(TableRepartition(
                TableFilterIntervals(c, intervals, keep),
                n,
                strategy,
              ))

            case TableLeftJoinRightDistinct(c, right, root) if keep =>
              Some(
                TableLeftJoinRightDistinct(
                  TableFilterIntervals(c, intervals, true),
                  TableFilterIntervals(right, intervals, true),
                  root,
                )
              )

            case TableIntervalJoin(c, right, root, product) =>
              Some(TableIntervalJoin(
                TableFilterIntervals(c, intervals, keep),
                right,
                root,
                product,
              ))

            case TableJoin(left, right, jt, jk) if keep =>
              Some(TableJoin(
                TableFilterIntervals(left, intervals, true),
                TableFilterIntervals(right, intervals, true),
                jt,
                jk,
              ))

            case TableExplode(c, path) =>
              Some(TableExplode(
                TableFilterIntervals(c, intervals, keep),
                path,
              ))

            case TableAggregateByKey(c, expr) =>
              Some(TableAggregateByKey(
                TableFilterIntervals(c, intervals, keep),
                expr,
              ))

            case TableFilterIntervals(c, _i1, keep1) if keep1 == keep =>
              val ord = PartitionBoundOrdering(
                ctx,
                c.typ.keyType,
              ).intervalEndpointOrdering
              val i1 = Interval.union(_i1, ord)
              val i2 = Interval.union(intervals, ord)
              val merged =
                if (keep1) Interval.intersection(i1, i2, ord) // keep means intersect intervals
                else Interval.union(i1 ++ i2, ord) // remove means union intervals
              Some(TableFilterIntervals(c, merged, keep1))

            // FIXME: Can try to serialize intervals shorter than the key
            // case TableKeyBy(child2, keys, isSorted) if !child2.typ.key.startsWith(keys) =>
            //   val ord = child.typ.keyType.ordering.intervalEndpointOrdering
            //   val maybeFlip: IR => IR = if (keep) identity else !_
            //   val pred =
            //     maybeFlip(invoke("sortedNonOverlappingIntervalsContain",
            //     TBoolean,
            // Literal(TArray(TInterval(k.typ.keyType)),
            //   Interval.union(intervals.toArray, ord).toFastIndexedSeq),
            //     MakeStruct(k.typ.keyType.fieldNames.map { keyField =>
            //       (keyField, GetField(Ref("row", child2.typ.rowType),
            //         keyField))
            //     })))
            //   TableKeyBy(TableFilter(child2, pred), keys, isSorted)

            case TableRead(t, false, tr: TableNativeReader)
                if keep
                  && tr.spec.indexed
                  && tr.params.options.forall(_.filterIntervals)
                  && SemanticVersion(
                    tr.spec.file_version
                  ) >= SemanticVersion(1, 3, 0) =>
              val newOpts = tr.params.options match {
                case None =>
                  val pt = t.keyType
                  NativeReaderOptions(
                    Interval.union(
                      intervals,
                      PartitionBoundOrdering(ctx, pt).intervalEndpointOrdering,
                    ),
                    pt,
                    true,
                  )
                case Some(NativeReaderOptions(preIntervals, intervalPointType, _)) =>
                  val iord = PartitionBoundOrdering(ctx, intervalPointType).intervalEndpointOrdering
                  NativeReaderOptions(
                    Interval.intersection(
                      Interval.union(preIntervals, iord),
                      Interval.union(intervals, iord),
                      iord,
                    ),
                    intervalPointType,
                    true,
                  )
              }

              Some(TableRead(
                t,
                false,
                new TableNativeReader(
                  TableNativeReaderParameters(tr.params.path, Some(newOpts)),
                  tr.spec,
                ),
              ))

            case TableRead(t, false, tr: TableNativeZippedReader)
                if keep
                  && tr.specLeft.indexed
                  && tr.options.forall(_.filterIntervals)
                  && SemanticVersion(tr.specLeft.file_version) >= SemanticVersion(1, 3, 0) =>
              val newOpts = tr.options match {
                case None =>
                  val pt = t.keyType
                  NativeReaderOptions(
                    Interval.union(
                      intervals,
                      PartitionBoundOrdering(ctx, pt).intervalEndpointOrdering,
                    ),
                    pt,
                    true,
                  )
                case Some(NativeReaderOptions(
                      preIntervals,
                      intervalPointType,
                      _,
                    )) =>
                  val iord = PartitionBoundOrdering(ctx, intervalPointType).intervalEndpointOrdering
                  NativeReaderOptions(
                    Interval.intersection(
                      Interval.union(preIntervals, iord),
                      Interval.union(intervals, iord),
                      iord,
                    ),
                    intervalPointType,
                    true,
                  )
              }

              Some(TableRead(
                t,
                false,
                TableNativeZippedReader(
                  tr.pathLeft,
                  tr.pathRight,
                  Some(newOpts),
                  tr.specLeft,
                  tr.specRight,
                ),
              ))

            case _ =>
              None
          }

      case _ =>
        None
    }

  private[this] def matrixRules(ctx: ExecuteContext, mir: MatrixIR): Option[MatrixIR] =
    mir match {
      case MatrixMapRows(child, newRow) =>
        newRow match {
          case Ref(n, _) if n == MatrixIR.rowName =>
            Some(child)

          case _ =>
            child match {
              // Note: the following MMR and MMC fusing rules are much weaker than they could be.
              // If they contain aggregations but those aggregations that mention "row" / "sa"
              // but do not depend on the updated value, we should locally prune and fuse anyway.
              case MatrixMapRows(inner, newRow1)
                  if !Mentions.inAggOrScan(newRow, MatrixIR.rowName)
                    && !Exists.inIR(
                      newRow,
                      {
                        // Lowering produces invalid IR
                        case a: ApplyAggOp => a.initOpArgs.exists(Mentions(_, MatrixIR.rowName))
                        case _ => false
                      },
                    ) =>
                Some(
                  MatrixMapRows(
                    inner,
                    bindIR(newRow1) { uid =>
                      Subst(
                        newRow,
                        BindingEnv(
                          Env(MatrixIR.rowName -> uid),
                          agg = Some(Env.empty),
                          scan = Some(Env.empty),
                        ),
                      )
                    },
                  )
                )

              case _ => None
            }
        }

      case MatrixKeyRowsBy(MatrixKeyRowsBy(child, _, innerIsSorted), keys, isSorted) =>
        if (!isSorted) Some(MatrixKeyRowsBy(child, keys, false))
        else if (innerIsSorted) Some(MatrixKeyRowsBy(child, keys, true))
        else None

      case MatrixMapCols(child, newCol, newKey) =>
        newCol match {
          case Ref(n, _) if n == MatrixIR.colName && newKey.isEmpty =>
            Some(child)

          case _ =>
            child match {
              case MatrixMapCols(inner, newCol1, nk1)
                  if !Mentions.inAggOrScan(newCol, MatrixIR.colName) =>
                Some(
                  MatrixMapCols(
                    inner,
                    newCol1.bind { uid =>
                      Subst(
                        newCol,
                        BindingEnv(
                          eval = Env(MatrixIR.colName -> uid),
                          agg = Some(Env.empty),
                          scan = Some(Env.empty),
                        ),
                      )
                    },
                    newKey.orElse(nk1),
                  )
                )
              case _ => None
            }
        }

      case x @ MatrixMapEntries(child, newEntries) =>
        newEntries match {
          case Ref(n, _) if n == MatrixIR.entryName =>
            assert(child.typ == x.typ)
            Some(child)

          case _ =>
            child match {
              case MatrixMapEntries(inner, newEntries1) =>
                Some(
                  MatrixMapEntries(
                    inner,
                    bindIR(newEntries1)(uid =>
                      Subst(
                        newEntries,
                        BindingEnv(
                          Env(MatrixIR.entryName -> uid),
                          agg = Some(Env.empty),
                          scan = Some(Env.empty),
                        ),
                      )
                    ),
                  )
                )
              case _ => None
            }
        }

      case MatrixMapGlobals(child, newGlobals) =>
        newGlobals match {
          case Ref(n, _) if n == MatrixIR.globalName =>
            Some(child)

          case _ =>
            child match {
              case MatrixMapGlobals(inner, ng1) =>
                Some(
                  MatrixMapGlobals(
                    inner,
                    bindIR(ng1)(uid =>
                      Subst(newGlobals, BindingEnv(Env(MatrixIR.globalName -> uid)))
                    ),
                  )
                )

              case _ => None
            }
        }

      // flatten unions
      case MatrixUnionRows(children) if children.exists(_.isInstanceOf[MatrixUnionRows]) =>
        Some(
          MatrixUnionRows(children.flatMap {
            case u: MatrixUnionRows => u.childrenSeq
            case c => Some(c)
          })
        )

      case MatrixFilterRows(child, pred) =>
        pred match {
          case _: True =>
            Some(child)

          case _ =>
            child match {
              case MatrixRead(typ, dropCols, _, reader) =>
                pred match {
                  case _: False | _: NA =>
                    Some(MatrixRead(typ, dropCols, dropRows = true, reader))

                  case _ => None
                }

              case MatrixFilterRows(inner, pred1) =>
                Some(MatrixFilterRows(inner, pred1 && pred))

              case _ => None
            }
        }

      case MatrixFilterCols(child, pred) =>
        pred match {
          case _: True =>
            Some(child)

          case _ =>
            child match {
              case MatrixRead(typ, _, dropRows, reader) =>
                pred match {
                  case _: False | _: NA =>
                    Some(MatrixRead(typ, dropCols = true, dropRows, reader))

                  case _ => None
                }

              case MatrixFilterCols(inner, pred1) =>
                Some(MatrixFilterCols(inner, pred1 && pred))
              // push MatrixFilterCols through MatrixMapEntries / MatrixFilterEntries
              // so that column-reducing operations run before per-entry work
              case MatrixMapEntries(inner, newEntries) =>
                Some(MatrixMapEntries(MatrixFilterCols(inner, pred), newEntries))

              case MatrixFilterEntries(inner, entryPred) =>
                Some(MatrixFilterEntries(MatrixFilterCols(inner, pred), entryPred))

              case _ => None
            }
        }

      case MatrixFilterEntries(MatrixFilterEntries(inner, pred1), pred) =>
        Some(MatrixFilterEntries(inner, pred1 && pred))

      // bubble up MatrixColsHead node
      case MatrixColsHead(child, n) =>
        child match {
          case MatrixMapCols(inner, newCol, newKey) =>
            Some(MatrixMapCols(MatrixColsHead(inner, n), newCol, newKey))

          case MatrixMapEntries(inner, newEntries) =>
            Some(MatrixMapEntries(MatrixColsHead(inner, n), newEntries))

          case MatrixFilterEntries(inner, newEntries) =>
            Some(MatrixFilterEntries(MatrixColsHead(inner, n), newEntries))

          case MatrixKeyRowsBy(inner, keys, isSorted) =>
            Some(MatrixKeyRowsBy(MatrixColsHead(inner, n), keys, isSorted))

          case MatrixAggregateRowsByKey(inner, rowExpr, entryExpr) =>
            Some(MatrixAggregateRowsByKey(
              MatrixColsHead(inner, n),
              rowExpr,
              entryExpr,
            ))

          case MatrixChooseCols(inner, oldIndices) =>
            Some(
              if (oldIndices.length <= n) child
              else MatrixChooseCols(inner, oldIndices.take(n))
            )

          case MatrixColsHead(inner, n1) =>
            Some(if (n1 <= n) child else MatrixColsHead(inner, n))

          case MatrixFilterRows(inner, pred) =>
            Some(MatrixFilterRows(MatrixColsHead(inner, n), pred))

          case MatrixRead(t, dr, dc, r: MatrixRangeReader) =>
            Some(
              MatrixRead(
                t,
                dr,
                dc,
                MatrixRangeReader(
                  ctx,
                  r.params.nRows,
                  math.min(r.params.nCols, n),
                  r.params.nPartitions,
                ),
              )
            )

          case MatrixMapRows(inner, newRow)
              if !Mentions.inAggOrScan(newRow, MatrixIR.colName) =>
            Some(MatrixMapRows(MatrixColsHead(inner, n), newRow))

          case MatrixMapGlobals(inner, newGlobals) =>
            Some(MatrixMapGlobals(MatrixColsHead(inner, n), newGlobals))

          case MatrixAnnotateColsTable(inner, table, root) =>
            Some(MatrixAnnotateColsTable(
              MatrixColsHead(inner, n),
              table,
              root,
            ))

          case MatrixAnnotateRowsTable(inner, table, root, product) =>
            Some(MatrixAnnotateRowsTable(
              MatrixColsHead(inner, n),
              table,
              root,
              product,
            ))

          case MatrixRepartition(inner, nPar, strategy) =>
            Some(MatrixRepartition(MatrixColsHead(inner, n), nPar, strategy))

          case MatrixExplodeRows(inner, path) =>
            Some(MatrixExplodeRows(MatrixColsHead(inner, n), path))

          case MatrixUnionRows(children) =>
            // could prevent a dimension mismatch error, but we view errors as undefined
            // behavior, so this seems OK.
            Some(MatrixUnionRows(children.map(MatrixColsHead(_, n))))

          case MatrixDistinctByRow(inner) =>
            Some(MatrixDistinctByRow(MatrixColsHead(inner, n)))

          case MatrixRename(inner, glob, col, row, entry) =>
            Some(MatrixRename(
              MatrixColsHead(inner, n),
              glob,
              col,
              row,
              entry,
            ))

          case _ => None
        }

      case _ =>
        None
    }

  private[this] def blockMatrixRules(ctx: ExecuteContext, bmir: BlockMatrixIR)
    : Option[BlockMatrixIR] =
    bmir match {
      case BlockMatrixBroadcast(child, IndexedSeq(0, 1), _, _) =>
        Some(child)

      case BlockMatrixSlice(BlockMatrixMap(child, n, f, reqDense), slices) =>
        Some(BlockMatrixMap(BlockMatrixSlice(child, slices), n, f, reqDense))
      case BlockMatrixSlice(BlockMatrixMap2(l, r, ln, rn, f, sparsityStrategy), slices) =>
        Some(BlockMatrixMap2(
          BlockMatrixSlice(l, slices),
          BlockMatrixSlice(r, slices),
          ln,
          rn,
          f,
          sparsityStrategy,
        ))
      case BlockMatrixMap2(
            BlockMatrixBroadcast(scalarBM, IndexedSeq(), _, _),
            right,
            leftName,
            rightName,
            f,
            sparsityStrategy,
          ) =>
        val needsDense = sparsityStrategy == NeedsDense || sparsityStrategy.exists(
          leftBlock = true,
          rightBlock = false,
        )
        val maybeDense = if (needsDense) BlockMatrixDensify(right) else right
        Some(
          BlockMatrixMap(
            maybeDense,
            rightName,
            BlockMatrixToValueApply(scalarBM, functions.GetElement(FastSeq(0L, 0L)))
              .bind(elem => Subst(f, BindingEnv.eval(leftName -> elem))),
            needsDense,
          )
        )
      case BlockMatrixMap2(
            left,
            BlockMatrixBroadcast(scalarBM, IndexedSeq(), _, _),
            leftName,
            rightName,
            f,
            sparsityStrategy,
          ) =>
        val needsDense = sparsityStrategy == NeedsDense || sparsityStrategy.exists(
          leftBlock = false,
          rightBlock = true,
        )
        val maybeDense = if (needsDense) BlockMatrixDensify(left) else left
        Some(
          BlockMatrixMap(
            maybeDense,
            leftName,
            BlockMatrixToValueApply(scalarBM, functions.GetElement(FastSeq(0L, 0L)))
              .bind(elem => Subst(f, BindingEnv.eval(rightName -> elem))),
            needsDense,
          )
        )
      case BlockMatrixMap(matrix, name, Ref(x, _), _) if name == x =>
        Some(matrix)
      case BlockMatrixMap(matrix, name, ir, _)
          if IsConstant(ir) || (ir.isInstanceOf[Ref] && ir.asInstanceOf[Ref].name != name) =>
        val typ = matrix.typ
        Some(BlockMatrixBroadcast(
          ValueToBlockMatrix(ir, FastSeq(1L, 1L), typ.blockSize),
          FastSeq(),
          FastSeq(typ.nRows, typ.nCols),
          typ.blockSize,
        ))
      case _ =>
        None
    }

  // Match on expressions of the form
  //  MakeStruct(IndexedSeq(a -> GetField(o, x) [, b -> GetField(o, y), ...]))
  // where
  //  - all fields are extracted from the same object, `o`
  //  - all references to the fields in o are unique
  private object MakeStructOfGetField {
    def unapply(ir: IR): Option[(IR, IndexedSeq[(String, String)])] =
      ir match {
        case MakeStruct(fields) if fields.nonEmpty =>
          val names = mutable.HashSet.empty[String]
          val rewrites = ArraySeq.newBuilder[(String, String)]
          rewrites.sizeHint(fields.length)

          fields.view.map {
            case (a, GetField(o, b)) if names.add(b) =>
              rewrites += (b -> a)
              Some(o)
            case _ => None
          }
            .reduce((a, b) => if (a == b) a else None)
            .map(_ -> rewrites.result())
        case _ =>
          None
      }
  }
}

package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.collection.implicits.{toRichIndexedSeq, toRichMap}
import is.hail.expr.ir.{Memoized => M, _}
import is.hail.expr.ir.analyses.{ColumnCount, PartitionCounts, PartitionCountsOrColumnCount}
import is.hail.expr.ir.defs._
import is.hail.io.bgen.MatrixBGENReader
import is.hail.rvd.PartitionBoundOrdering
import is.hail.types.tcoerce
import is.hail.types.virtual._
import is.hail.utils._

import scala.collection.mutable

object Simplify {

  /** Transform 'ir' using simplification rules until none apply. */
  def apply(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    TimedBlock.enter(recur(ctx, ir))

  private[this] def recur(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    ir match {
      case ir: Atom => ir
      case ir: IR => simplifyValue(ctx, ir)
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

  implicit private[this] class RuleOps[A](private val f: A => Option[A]) extends AnyVal {
    def <>(g: A => Option[A]): A => Option[A] =
      a =>
        f(a)
          .map(improved => g(improved).getOrElse(improved))
          .orElse(g(a))
  }

  private[this] val simplifyValue: (ExecuteContext, IR) => IR =
    visitNode((_, ir) => (valueRules _ <> numericRules)(ir), simplifyValue)

  private[this] val simplifyTable: (ExecuteContext, TableIR) => TableIR =
    visitNode(tableRules, simplifyTable)

  private[this] val simplifyMatrix: (ExecuteContext, MatrixIR) => MatrixIR =
    visitNode(matrixRules, simplifyMatrix)

  private[this] val simplifyBlockMatrix: (ExecuteContext, BlockMatrixIR) => BlockMatrixIR =
    visitNode(blockMatrixRules, simplifyBlockMatrix)

  /** Returns true if any strict child of 'x' is NA. A child is strict if 'x' evaluates to missing
    * whenever the child does.
    */
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

  /** Returns true if 'x' will never evaluate to missing. */
  private[this] def isDefinitelyDefined(x: IR): Boolean = {
    x match {
      case _: MakeArray |
          _: MakeStruct |
          _: MakeTuple |
          _: IsNA |
          ApplyComparisonOp(EQWithNA, _, _) |
          ApplyComparisonOp(NEQWithNA, _, _) |
          _: I32 | _: I64 | _: F32 | _: F64 | True() | False() => true
      case _ => false
    }
  }

  private[this] def numericRules(ir: IR): Option[IR] =
    if (!ir.typ.isPrimitive) None
    else {
      val typ = ir.typ

      def isIntegral(t: Type) = t.isInstanceOf[TIntegral]
      def isFloating(t: Type) = t == TFloat32 || t == TFloat64

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

        case ApplyComparisonOp(op, x, y)
            if (!isFloating(x.typ) && x.typ == y.typ) && x == y =>
          op match {
            case LT => Some(False())
            case LTEQ => Some(True())
            case EQ => Some(True())
            case GTEQ => Some(True())
            case GT => Some(False())
            case NEQ => Some(False())
            case _ => None
          }

        case ApplyUnaryPrimOp(f @ (Negate | BitNot | Bang), x) =>
          x match {
            case ApplyUnaryPrimOp(g, y) if g == f => Some(y)
            case _ => None
          }

        case _ =>
          None
      }
    }

  private[this] def valueRules(ir: IR): Option[IR] =
    ir match {
      // propagate NA
      case x: IR if hasMissingStrictChild(x) =>
        Some(NA(x.typ))

      case Coalesce(values) =>
        if (values.length == 1) Some(values.head)
        else {
          val idx = values.indexWhere(isDefinitelyDefined)
          if (idx >= 0 && idx < values.length - 1) Some(Coalesce(values.take(idx + 1)))
          else None
        }

      case IsNA(a) =>
        a match {
          case _: NA => Some(True())
          case _ if isDefinitelyDefined(a) => Some(False())
          case _ => None
        }

      case If(pred, cnsq, altr) =>
        pred match {
          case _: NA =>
            Some(NA(ir.typ))

          case _: True =>
            Some(cnsq)

          case _: False =>
            Some(altr)

          case ApplyUnaryPrimOp(Bang, c) =>
            Some(If(c, altr, cnsq))

          case _ if cnsq == altr && cnsq.typ != TVoid =>
            if (isDefinitelyDefined(pred)) Some(cnsq)
            else Some(If(IsNA(pred), NA(cnsq.typ), cnsq))

          case _ =>
            cnsq match {
              case If(c2, cnsq2, _) if pred == c2 =>
                Some(If(pred, cnsq2, altr))

              case _ =>
                altr match {
                  case If(c2, _, altr2) if pred == c2 =>
                    Some(If(pred, cnsq, altr2))

                  case _ =>
                    ir match {
                      case If(IsNA(a), NA(_), b) if a == b =>
                        Some(b)
                      case _ =>
                        None
                    }
                }
            }
        }

      case Switch(x, default, cases) =>
        x match {
          case I32(v) => Some(if (v >= 0 && v < cases.length) cases(v) else default)
          case _ if cases.isEmpty && isDefinitelyDefined(x) => Some(default)
          case _ => None
        }

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

      case ArrayLen(child) =>
        child match {
          case MakeArray(args, _) => Some(I32(args.length))
          case ToArray(s) => Some(StreamLen(s))
          case StreamFlatMap(a, _, MakeArray(args, _)) => Some(I32(args.length) * ArrayLen(a))
          case ArraySort(a, _, _, _) => Some(ArrayLen(ToArray(a)))
          case ToArray(MakeStream(args, _, _)) => Some(I32(args.length))
          case GetField(TableCollect(tc), "rows") => Some(Cast(TableCount(tc), TInt32))
          case _ => None
        }

      case ArraySlice(ToArray(s), start, stop, step, _) =>
        (start, stop, step) match {
          case (I32(0), Some(x @ I32(i)), I32(1)) if i >= 0 =>
            Some(ToArray(StreamTake(s, x)))

          case (x @ I32(i), Some(I32(j)), I32(1)) if i > 0 && j > 0 =>
            Some(
              if (j > i) ToArray(StreamTake(StreamDrop(s, x), I32(j - i)))
              else MakeArray.empty(TIterable.elementType(s.typ))
            )

          case (x @ I32(i), None, I32(1)) if i >= 0 =>
            Some(ToArray(StreamDrop(s, x)))

          case _ =>
            None
        }

      case ArrayRef(MakeArray(args, _), I32(i), _) if i >= 0 && i < args.length =>
        Some(args(i))

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
      case StreamFold(stream, zero, accum, elem, f) =>
        stream match {
          case Block(bindings, body) =>
            Some(Block(bindings, StreamFold(body, zero, accum, elem, f)))

          case StreamMap(a, name, g) if a.typ.asInstanceOf[TStream].elementType.isRealizable =>
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

          case _: NA =>
            Some(NA(ir.typ))

          case _ =>
            cond match {
              case True() => Some(stream)
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

          case _ =>
            None
        }

      case StreamMap(stream, x, f) =>
        stream match {
          case Block(bindings, body) =>
            Some(Block(bindings, StreamMap(body, x, f)))

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

      case CastToArray(x) if x.typ.isInstanceOf[TArray] =>
        Some(x)

      case ToArray(child) =>
        child match {
          case ToStream(a, _) if a.typ.isInstanceOf[TArray] =>
            Some(a)

          case ToStream(a, _) if a.typ.isInstanceOf[TSet] || a.typ.isInstanceOf[TDict] =>
            Some(CastToArray(a))

          case _ =>
            None
        }

      case ToStream(child, memManaged) if !memManaged =>
        child match {
          case ToArray(s) if s.typ.isInstanceOf[TStream] =>
            Some(s)
          case Block(bindings, ToArray(x)) if x.typ.isInstanceOf[TStream] =>
            Some(Block(bindings, x))

          case _ =>
            None
        }

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
            else None
        }

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
                    newBindings ++= (if (scope == Scope.EVAL) ir.bindings
                                     else ir.bindings.map {
                                       case Binding(name, value, Scope.EVAL) =>
                                         Binding(name, value, scope)
                                       case _ => fatal("Simplify: found nested Agg bindings")
                                     })
                    newBindings += Binding(name, ir.body, scope)
                  case binding => newBindings += binding
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
            None
        }

      case MakeStructOfGetField(o, newNames) =>
        val select = SelectFields(o, newNames.map(_._1))
        Some(CastRename(select, select.typ.rename(newNames.toMap)))

      case GetTupleElement(MakeTuple(xs), idx) =>
        Some(xs.find(_._1 == idx).get._2)

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
              Subst(newGlobals, BindingEnv(Env.empty[IR].bind(TableIR.globalName, ref)))
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

      /* NOTE: The below rule should be reintroduced when it is possible to put an ArrayAgg inside a
       * TableAggregate */
      // case TableAggregate(TableParallelize(rowsAndGlobal, _), query) =>
      //   rowsAndGlobal match {
      /* // match because we currently don't optimize MakeStruct through Let, and this is a common
       * pattern */
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

      case ApplyComparisonOp(EQ, l, r) =>
        (l, r) match {
          case (_, True()) => Some(l)
          case (True(), _) => Some(r)
          case (_, False()) => Some(ApplyUnaryPrimOp(Bang, l))
          case (False(), _) => Some(ApplyUnaryPrimOp(Bang, r))
          case _ => None
        }

      case ApplyUnaryPrimOp(Bang, ApplyComparisonOp(op, l, r)) =>
        Some(ApplyComparisonOp(ComparisonOp.negate(op.asInstanceOf[ComparisonOp[Boolean]]), l, r))

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

  private[this] def tableRules(ctx: ExecuteContext, tir: TableIR): Option[TableIR] =
    tir match {
      case TableRename(child, m1, m2) if m1.isTrivial && m2.isTrivial =>
        Some(child)

      // TODO: Write more rules like this to bubble 'TableRename' nodes towards the root.
      case t @ TableRename(TableKeyBy(child, keys, isSorted, nPartitions), rowMap, globalMap) =>
        Some(TableKeyBy(
          TableRename(child, rowMap, globalMap),
          keys.map(t.rowF),
          isSorted,
          nPartitions,
        ))

      case TableFilter(t, True()) =>
        Some(t)

      case TableFilter(TableRead(typ, _, tr), False() | NA(_)) =>
        Some(TableRead(typ, dropRows = true, tr))

      case TableFilter(TableFilter(t, p1), p2) =>
        Some(TableFilter(
          t,
          ApplySpecial("land", ArraySeq.empty, ArraySeq(p1, p2), TBoolean, ErrorIDs.NO_ERROR),
        ))

      case TableFilter(TableKeyBy(child, key, isSorted, nPartitions), p) =>
        Some(TableKeyBy(TableFilter(child, p), key, isSorted, nPartitions))

      case TableFilter(TableRepartition(child, n, strategy), p) =>
        Some(TableRepartition(TableFilter(child, p), n, strategy))

      case TableOrderBy(TableKeyBy(child, _, false, _), sortFields) =>
        Some(TableOrderBy(child, sortFields))

      case TableFilter(TableOrderBy(child, sortFields), pred) =>
        Some(TableOrderBy(TableFilter(child, pred), sortFields))

      case TableFilter(TableParallelize(rowsAndGlobal, nPartitions), pred) =>
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
                  struct.get("global").map(mkFilteredRowsAndGlobal(struct.get("rows"), _))
                }
              }
          }

        Some(TableParallelize(newRowsAndGlobal, nPartitions))

      case TableKeyBy(TableOrderBy(child, _), keys, false, nPartitions) =>
        Some(TableKeyBy(child, keys, false, nPartitions))

      case TableKeyBy(TableKeyBy(child, _, _, _), keys, false, nPartitions) =>
        Some(TableKeyBy(child, keys, false, nPartitions))

      case TableKeyBy(TableKeyBy(child, _, true, _), keys, true, nPartitions) =>
        Some(TableKeyBy(child, keys, true, nPartitions))

      case TableKeyBy(child, key, _, _) if key == child.typ.key =>
        Some(child)

      case TableMapRows(child, Ref(n, _)) if n == TableIR.rowName =>
        Some(child)

      case TableMapRows(child, MakeStruct(fields))
          if fields.length == child.typ.rowType.size
            && fields.zip(child.typ.rowType.fields).forall { case ((_, ir), field) =>
              ir == GetField(Ref(TableIR.rowName, field.typ), field.name)
            } =>
        val renamedPairs = for {
          (oldName, (newName, _)) <- child.typ.rowType.fieldNames zip fields
          if oldName != newName
        } yield oldName -> newName
        Some(TableRename(child, Map(renamedPairs: _*), Map.empty))

      case TableMapRows(TableMapRows(child, f), g) if !ContainsScan(g) =>
        val newRow = f.bind(r => Subst(g, BindingEnv.eval(TableIR.rowName -> r)))
        Some(TableMapRows(child, newRow))

      case TableMapGlobals(child, Ref(n, _)) if n == TableIR.globalName =>
        Some(child)

      // flatten unions
      case TableUnion(children) if children.exists(_.isInstanceOf[TableUnion]) =>
        Some(TableUnion(children.flatMap {
          case u: TableUnion => u.childrenSeq
          case c => Some(c)
        }))

      case MatrixRowsTable(MatrixUnionRows(children)) =>
        Some(TableUnion(children.map(MatrixRowsTable)))

      case MatrixColsTable(MatrixUnionRows(children)) =>
        Some(MatrixColsTable(children(0)))

      // Ignore column or row data that is immediately dropped
      case MatrixRowsTable(MatrixRead(typ, false, dropRows, reader)) =>
        Some(MatrixRowsTable(MatrixRead(typ, dropCols = true, dropRows, reader)))

      case MatrixColsTable(MatrixRead(typ, dropCols, false, reader)) =>
        Some(MatrixColsTable(MatrixRead(typ, dropCols, dropRows = true, reader)))

      case MatrixRowsTable(MatrixFilterRows(child, pred)) =>
        val mrt = MatrixRowsTable(child)
        Some(TableFilter(
          mrt,
          Subst(pred, BindingEnv(Env(MatrixIR.rowName -> Ref(TableIR.rowName, mrt.typ.rowType)))),
        ))

      case MatrixRowsTable(MatrixMapGlobals(child, newGlobals)) =>
        Some(TableMapGlobals(MatrixRowsTable(child), newGlobals))

      case MatrixRowsTable(MatrixMapCols(child, _, _)) =>
        Some(MatrixRowsTable(child))

      case MatrixRowsTable(MatrixMapEntries(child, _)) =>
        Some(MatrixRowsTable(child))

      case MatrixRowsTable(MatrixFilterEntries(child, _)) =>
        Some(MatrixRowsTable(child))

      case MatrixRowsTable(MatrixFilterCols(child, _)) =>
        Some(MatrixRowsTable(child))

      case MatrixRowsTable(MatrixAggregateColsByKey(child, _, _)) =>
        Some(MatrixRowsTable(child))

      case MatrixRowsTable(MatrixChooseCols(child, _)) =>
        Some(MatrixRowsTable(child))

      case MatrixRowsTable(MatrixCollectColsByKey(child)) =>
        Some(MatrixRowsTable(child))

      case MatrixRowsTable(MatrixKeyRowsBy(child, keys, isSorted)) =>
        Some(TableKeyBy(MatrixRowsTable(child), keys, isSorted))

      case MatrixColsTable(MatrixMapCols(child, newRow, newKey))
          if newKey.isEmpty
            && !ContainsAgg(newRow)
            && !ContainsScan(newRow) =>
        val mct = MatrixColsTable(child)
        Some(TableMapRows(
          mct,
          Subst(newRow, BindingEnv(Env(MatrixIR.colName -> Ref(TableIR.rowName, mct.typ.rowType)))),
        ))

      case MatrixColsTable(MatrixMapGlobals(child, newGlobals)) =>
        Some(TableMapGlobals(MatrixColsTable(child), newGlobals))

      case MatrixColsTable(MatrixMapRows(child, _)) =>
        Some(MatrixColsTable(child))

      case MatrixColsTable(MatrixMapEntries(child, _)) =>
        Some(MatrixColsTable(child))

      case MatrixColsTable(MatrixFilterEntries(child, _)) =>
        Some(MatrixColsTable(child))

      case MatrixColsTable(MatrixFilterRows(child, _)) =>
        Some(MatrixColsTable(child))

      case MatrixColsTable(MatrixAggregateRowsByKey(child, _, _)) =>
        Some(MatrixColsTable(child))

      case MatrixColsTable(MatrixKeyRowsBy(child, _, _)) =>
        Some(MatrixColsTable(child))

      case TableRepartition(TableRange(nRows, _), nParts, _) =>
        Some(TableRange(nRows, nParts))

      case TableMapGlobals(TableMapGlobals(child, ng1), ng2) =>
        Some(TableMapGlobals(
          child,
          bindIR(ng1)(uid => Subst(ng2, BindingEnv(Env(TableIR.globalName -> uid)))),
        ))

      case TableHead(MatrixColsTable(child), n) if child.typ.colKey.isEmpty =>
        Some(
          if (n > Int.MaxValue) MatrixColsTable(child)
          else MatrixColsTable(MatrixColsHead(child, n.toInt))
        )

      case TableHead(TableMapRows(child, newRow), n) =>
        Some(TableMapRows(TableHead(child, n), newRow))

      case TableHead(TableRepartition(child, nPar, shuffle), n) =>
        Some(TableRepartition(TableHead(child, n), nPar, shuffle))

      case TableHead(tr @ TableRange(nRows, nPar), n) =>
        Some(
          if (n < nRows) TableRange(n.toInt, (nPar.toFloat * n / nRows).toInt.max(1))
          else tr
        )

      case TableHead(TableMapGlobals(child, newGlobals), n) =>
        Some(TableMapGlobals(TableHead(child, n), newGlobals))

      case TableHead(TableOrderBy(child, sortFields), n)
          if !TableOrderBy.isAlreadyOrdered(
            sortFields,
            child.typ.key,
          ) // FIXME: https://github.com/hail-is/hail/issues/6234
            && sortFields.forall(_.sortOrder == Ascending)
            && n < 256 =>
        // n < 256 is arbitrary for memory concerns
        Some(
          child
            .keyByAndAggregate(10, Some(1)) { (_, row) =>
              val keyStruct = MakeStruct(sortFields.map(f => f.field -> row.get(f.field)))
              makestruct("__row" -> ApplyAggOp(TakeBy(), n.toInt)(row, keyStruct))
            }((_, _) => makestruct())
            .explode("__row")
            .mapRows((_, row) => row.get("__row"))
        )

      case TableDistinct(TableDistinct(child)) =>
        Some(TableDistinct(child))

      case TableDistinct(TableAggregateByKey(child, expr)) =>
        Some(TableAggregateByKey(child, expr))

      case TableDistinct(TableMapRows(child, newRow)) =>
        Some(TableMapRows(TableDistinct(child), newRow))

      case TableDistinct(TableLeftJoinRightDistinct(child, right, root)) =>
        Some(TableLeftJoinRightDistinct(TableDistinct(child), right, root))

      case TableDistinct(TableRepartition(child, n, strategy)) =>
        Some(TableRepartition(TableDistinct(child), n, strategy))

      case TableKeyByAndAggregate(child, MakeStruct(Seq()), k @ MakeStruct(_), _, _) =>
        Some(
          TableDistinct(TableKeyBy(
            TableMapRows(TableKeyBy(child, FastSeq()), k),
            k.typ.fieldNames,
          ))
        )

      case TableKeyByAndAggregate(child, expr, newKey, _, _)
          if (newKey == MakeStruct(child.typ.key.map(k =>
            k -> GetField(Ref(TableIR.rowName, child.typ.rowType), k)
          )) ||
            newKey == SelectFields(Ref(TableIR.rowName, child.typ.rowType), child.typ.key))
            && child.typ.key.nonEmpty =>
        Some(TableAggregateByKey(child, expr))

      case TableAggregateByKey(x @ TableKeyBy(child, keys, false, nPartitions), expr)
          if !x.definitelyDoesNotShuffle =>
        Some(TableKeyByAndAggregate(
          child,
          expr,
          MakeStruct(keys.map(k => k -> GetField(Ref(TableIR.rowName, child.typ.rowType), k))),
          bufferSize = ctx.getFlag("grouped_aggregate_buffer_size").toInt,
          nPartitions = nPartitions,
        ))

      case TableParallelize(TableCollect(child), _) =>
        Some(child)

      case TableFilterIntervals(child, intervals, keep) if intervals.isEmpty =>
        if (keep) Some(TableFilter(child, False()))
        else Some(child)

      // push down filter intervals nodes
      case TableFilterIntervals(TableFilter(child, pred), intervals, keep) =>
        Some(TableFilter(TableFilterIntervals(child, intervals, keep), pred))

      case TableFilterIntervals(TableMapRows(child, newRow), intervals, keep)
          if !ContainsScan(newRow) =>
        Some(TableMapRows(TableFilterIntervals(child, intervals, keep), newRow))

      case TableFilterIntervals(TableMapGlobals(child, newRow), intervals, keep) =>
        Some(TableMapGlobals(TableFilterIntervals(child, intervals, keep), newRow))

      case TableFilterIntervals(TableRename(child, rowMap, globalMap), intervals, keep) =>
        Some(TableRename(TableFilterIntervals(child, intervals, keep), rowMap, globalMap))

      case TableFilterIntervals(TableRepartition(child, n, strategy), intervals, keep) =>
        Some(TableRepartition(TableFilterIntervals(child, intervals, keep), n, strategy))

      case TableFilterIntervals(TableLeftJoinRightDistinct(child, right, root), intervals, true) =>
        Some(
          TableLeftJoinRightDistinct(
            TableFilterIntervals(child, intervals, true),
            TableFilterIntervals(right, intervals, true),
            root,
          )
        )

      case TableFilterIntervals(TableIntervalJoin(child, right, root, product), intervals, keep) =>
        Some(TableIntervalJoin(TableFilterIntervals(child, intervals, keep), right, root, product))

      case TableFilterIntervals(TableJoin(left, right, jt, jk), intervals, true) =>
        Some(TableJoin(
          TableFilterIntervals(left, intervals, true),
          TableFilterIntervals(right, intervals, true),
          jt,
          jk,
        ))

      case TableFilterIntervals(TableExplode(child, path), intervals, keep) =>
        Some(TableExplode(TableFilterIntervals(child, intervals, keep), path))

      case TableFilterIntervals(TableAggregateByKey(child, expr), intervals, keep) =>
        Some(TableAggregateByKey(TableFilterIntervals(child, intervals, keep), expr))
      case TableFilterIntervals(TableFilterIntervals(child, _i1, keep1), _i2, keep2)
          if keep1 == keep2 =>
        val ord = PartitionBoundOrdering(ctx, child.typ.keyType).intervalEndpointOrdering
        val i1 = Interval.union(_i1, ord)
        val i2 = Interval.union(_i2, ord)
        val intervals = if (keep1)
          // keep means intersect intervals
          Interval.intersection(i1, i2, ord)
        else
          // remove means union intervals
          Interval.union(i1 ++ i2, ord)
        Some(TableFilterIntervals(child, intervals, keep1))

      // FIXME: Can try to serialize intervals shorter than the key
      /* case TableFilterIntervals(k@TableKeyBy(child, keys, isSorted), intervals, keep) if
       * !child.typ.key.startsWith(keys) => */
      //   val ord = k.typ.keyType.ordering.intervalEndpointOrdering
      //   val maybeFlip: IR => IR = if (keep) identity else !_
      //   val pred = maybeFlip(invoke("sortedNonOverlappingIntervalsContain",
      //     TBoolean,
      /* Literal(TArray(TInterval(k.typ.keyType)), Interval.union(intervals.toArray,
       * ord).toFastIndexedSeq), */
      //     MakeStruct(k.typ.keyType.fieldNames.map { keyField =>
      //       (keyField, GetField(Ref("row", child.typ.rowType), keyField))
      //     })))
      //   TableKeyBy(TableFilter(child, pred), keys, isSorted)

      case TableFilterIntervals(TableRead(t, false, tr: TableNativeReader), intervals, true)
          if tr.spec.indexed
            && tr.params.options.forall(_.filterIntervals)
            && SemanticVersion(tr.spec.file_version) >= SemanticVersion(1, 3, 0) =>
        val newOpts = tr.params.options match {
          case None =>
            val pt = t.keyType
            NativeReaderOptions(
              Interval.union(intervals, PartitionBoundOrdering(ctx, pt).intervalEndpointOrdering),
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
          new TableNativeReader(TableNativeReaderParameters(tr.params.path, Some(newOpts)), tr.spec),
        ))

      case TableFilterIntervals(TableRead(t, false, tr: TableNativeZippedReader), intervals, true)
          if tr.specLeft.indexed
            && tr.options.forall(_.filterIntervals)
            && SemanticVersion(tr.specLeft.file_version) >= SemanticVersion(1, 3, 0) =>
        val newOpts = tr.options match {
          case None =>
            val pt = t.keyType
            NativeReaderOptions(
              Interval.union(intervals, PartitionBoundOrdering(ctx, pt).intervalEndpointOrdering),
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

  private[this] def matrixRules(ctx: ExecuteContext, mir: MatrixIR): Option[MatrixIR] =
    mir match {
      case MatrixMapRows(child, Ref(n, _)) if n == MatrixIR.rowName =>
        Some(child)

      case MatrixKeyRowsBy(MatrixKeyRowsBy(child, _, _), keys, false) =>
        Some(MatrixKeyRowsBy(child, keys, false))

      case MatrixKeyRowsBy(MatrixKeyRowsBy(child, _, true), keys, true) =>
        Some(MatrixKeyRowsBy(child, keys, true))

      case MatrixMapCols(child, Ref(n, _), None) if n == MatrixIR.colName =>
        Some(child)

      case x @ MatrixMapEntries(child, Ref(n, _)) if n == MatrixIR.entryName =>
        assert(child.typ == x.typ)
        Some(child)

      case MatrixMapEntries(MatrixMapEntries(child, newEntries1), newEntries2) =>
        Some(
          MatrixMapEntries(
            child,
            bindIR(newEntries1)(uid =>
              Subst(newEntries2, BindingEnv(Env(MatrixIR.entryName -> uid)))
            ),
          )
        )

      case MatrixMapGlobals(child, Ref(n, _)) if n == MatrixIR.globalName =>
        Some(child)

      // flatten unions
      case MatrixUnionRows(children) if children.exists(_.isInstanceOf[MatrixUnionRows]) =>
        Some(
          MatrixUnionRows(children.flatMap {
            case u: MatrixUnionRows => u.childrenSeq
            case c => Some(c)
          })
        )

      // Equivalent rewrites for the new Filter{Cols,Rows}IR
      case MatrixFilterRows(MatrixRead(typ, dropCols, _, reader), False() | NA(_)) =>
        Some(MatrixRead(typ, dropCols, dropRows = true, reader))

      case MatrixFilterCols(MatrixRead(typ, _, dropRows, reader), False() | NA(_)) =>
        Some(MatrixRead(typ, dropCols = true, dropRows, reader))

      // Keep all rows/cols = do nothing
      case MatrixFilterRows(m, True()) =>
        Some(m)

      case MatrixFilterCols(m, True()) =>
        Some(m)

      case MatrixFilterRows(MatrixFilterRows(child, pred1), pred2) =>
        Some(
          MatrixFilterRows(
            child,
            ApplySpecial("land", FastSeq(), FastSeq(pred1, pred2), TBoolean, ErrorIDs.NO_ERROR),
          )
        )

      case MatrixFilterCols(MatrixFilterCols(child, pred1), pred2) =>
        Some(
          MatrixFilterCols(
            child,
            ApplySpecial("land", FastSeq(), FastSeq(pred1, pred2), TBoolean, ErrorIDs.NO_ERROR),
          )
        )

      // push MatrixFilterCols through MatrixMapEntries / MatrixFilterEntries
      // so that column-reducing operations run before per-entry work
      case MatrixFilterCols(MatrixMapEntries(child, newEntries), pred) =>
        Some(MatrixMapEntries(MatrixFilterCols(child, pred), newEntries))

      case MatrixFilterCols(MatrixFilterEntries(child, entryPred), colPred) =>
        Some(MatrixFilterEntries(MatrixFilterCols(child, colPred), entryPred))

      case MatrixFilterEntries(MatrixFilterEntries(child, pred1), pred2) =>
        Some(MatrixFilterEntries(
          child,
          ApplySpecial("land", FastSeq(), FastSeq(pred1, pred2), TBoolean, ErrorIDs.NO_ERROR),
        ))

      case MatrixMapGlobals(MatrixMapGlobals(child, ng1), ng2) =>
        Some(
          MatrixMapGlobals(
            child,
            bindIR(ng1)(uid => Subst(ng2, BindingEnv(Env(MatrixIR.globalName -> uid)))),
          )
        )

      /* Note: the following MMR and MMC fusing rules are much weaker than they could be. If they
       * contain aggregations but those aggregations that mention "row" / "sa" but do not depend on
       * the updated value, we should locally prune and fuse anyway. */
      case MatrixMapRows(MatrixMapRows(child, newRow1), newRow2)
          if !Mentions.inAggOrScan(newRow2, MatrixIR.rowName)
            && !Exists.inIR(
              newRow2,
              {
                case a: ApplyAggOp =>
                  a.initOpArgs.exists(Mentions(_, MatrixIR.rowName)) // Lowering produces invalid IR
                case _ => false
              },
            ) =>
        Some(
          MatrixMapRows(
            child,
            bindIR(newRow1) { uid =>
              Subst(
                newRow2,
                BindingEnv[IR](
                  Env(MatrixIR.rowName -> uid),
                  agg = Some(Env.empty[IR]),
                  scan = Some(Env.empty[IR]),
                ),
              )
            },
          )
        )

      case MatrixMapCols(MatrixMapCols(child, newCol1, nk1), newCol2, nk2)
          if !Mentions.inAggOrScan(newCol2, MatrixIR.colName) =>
        Some(
          MatrixMapCols(
            child,
            bindIR(newCol1) { uid =>
              Subst(
                newCol2,
                BindingEnv[IR](
                  Env(MatrixIR.colName -> uid),
                  agg = Some(Env.empty[IR]),
                  scan = Some(Env.empty[IR]),
                ),
              )
            },
            nk2.orElse(nk1),
          )
        )

      // bubble up MatrixColsHead node
      case MatrixColsHead(MatrixMapCols(child, newCol, newKey), n) =>
        Some(MatrixMapCols(MatrixColsHead(child, n), newCol, newKey))

      case MatrixColsHead(MatrixMapEntries(child, newEntries), n) =>
        Some(MatrixMapEntries(MatrixColsHead(child, n), newEntries))

      case MatrixColsHead(MatrixFilterEntries(child, newEntries), n) =>
        Some(MatrixFilterEntries(MatrixColsHead(child, n), newEntries))

      case MatrixColsHead(MatrixKeyRowsBy(child, keys, isSorted), n) =>
        Some(MatrixKeyRowsBy(MatrixColsHead(child, n), keys, isSorted))

      case MatrixColsHead(MatrixAggregateRowsByKey(child, rowExpr, entryExpr), n) =>
        Some(MatrixAggregateRowsByKey(MatrixColsHead(child, n), rowExpr, entryExpr))

      case MatrixColsHead(MatrixChooseCols(child, oldIndices), n) =>
        Some(MatrixChooseCols(child, oldIndices.take(n)))

      case MatrixColsHead(MatrixColsHead(child, n1), n2) =>
        Some(MatrixColsHead(child, math.min(n1, n2)))

      case MatrixColsHead(MatrixFilterRows(child, pred), n) =>
        Some(MatrixFilterRows(MatrixColsHead(child, n), pred))

      case MatrixColsHead(MatrixRead(t, dr, dc, r: MatrixRangeReader), n) =>
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
      case MatrixColsHead(MatrixMapRows(child, newRow), n)
          if !Mentions.inAggOrScan(newRow, MatrixIR.colName) =>
        Some(MatrixMapRows(MatrixColsHead(child, n), newRow))

      case MatrixColsHead(MatrixMapGlobals(child, newGlobals), n) =>
        Some(MatrixMapGlobals(MatrixColsHead(child, n), newGlobals))

      case MatrixColsHead(MatrixAnnotateColsTable(child, table, root), n) =>
        Some(MatrixAnnotateColsTable(MatrixColsHead(child, n), table, root))

      case MatrixColsHead(MatrixAnnotateRowsTable(child, table, root, product), n) =>
        Some(MatrixAnnotateRowsTable(MatrixColsHead(child, n), table, root, product))

      case MatrixColsHead(MatrixRepartition(child, nPar, strategy), n) =>
        Some(MatrixRepartition(MatrixColsHead(child, n), nPar, strategy))

      case MatrixColsHead(MatrixExplodeRows(child, path), n) =>
        Some(MatrixExplodeRows(MatrixColsHead(child, n), path))

      case MatrixColsHead(MatrixUnionRows(children), n) =>
        /* could prevent a dimension mismatch error, but we view errors as undefined behavior, so
         * this seems OK. */
        Some(MatrixUnionRows(children.map(MatrixColsHead(_, n))))

      case MatrixColsHead(MatrixDistinctByRow(child), n) =>
        Some(MatrixDistinctByRow(MatrixColsHead(child, n)))

      case MatrixColsHead(MatrixRename(child, glob, col, row, entry), n) =>
        Some(MatrixRename(MatrixColsHead(child, n), glob, col, row, entry))

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

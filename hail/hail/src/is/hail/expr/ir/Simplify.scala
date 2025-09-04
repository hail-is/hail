package is.hail.expr.ir

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.analyses.{ColumnCount, PartitionCounts, PartitionCountsOrColumnCount}
import is.hail.expr.ir.defs._
import is.hail.io.bgen.MatrixBGENReader
import is.hail.rvd.PartitionBoundOrdering
import is.hail.types.tcoerce
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.utils.compat.immutable.ArraySeq

import scala.collection.mutable

object Simplify {

  /** Transform 'ir' using simplification rules until none apply. */
  def apply(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    ctx.time(recur(ctx, ir))

  private[this] def recur(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    ir match {
      case ir: TrivialIR => ir
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
          _: ApplySeeded |
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
      def pure(x: Int): IR = Literal.coerce(typ, x)

      lazy val MinusOne: IR = pure(-1)
      lazy val Zero: IR = pure(0)
      lazy val One: IR = pure(1)

      ir match {
        case ApplyBinaryPrimOp(op, x, y) =>
          op match {
            case Add() =>
              if (x == Zero) Some(y)
              else if (y == Zero) Some(x)
              else if (isIntegral(typ) && x == y) Some(ApplyBinaryPrimOp(Multiply(), pure(2), x))
              else None

            case Subtract() =>
              if (x == Zero) Some(ApplyUnaryPrimOp(Negate, y))
              else if (y == Zero) Some(x)
              else if (isIntegral(typ) && x == y) Some(Zero)
              else None

            case Multiply() =>
              if (x == One) Some(y)
              else if (x == MinusOne) Some(ApplyUnaryPrimOp(Negate, y))
              else if (y == One) Some(x)
              else if (y == MinusOne) Some(ApplyUnaryPrimOp(Negate, x))
              else if (isIntegral(typ) && (x == Zero || y == Zero)) Some(Zero)
              else None

            case RoundToNegInfDivide() =>
              if (y == One) Some(x)
              else if (y == MinusOne) Some(ApplyUnaryPrimOp(Negate, x))
              else if (isIntegral(typ)) {
                if (x == y) Some(One)
                else if (x == Zero) Some(Zero)
                else if (y == Zero) Some(Die("division by zero", ir.typ))
                else None
              } else None

            case _: LeftShift | _: RightShift | _: LogicalRightShift if isIntegral(typ) =>
              if (x == Zero) Some(Zero)
              else if (y == I32(0)) Some(x)
              else None

            case BitAnd() if isIntegral(typ) =>
              if (x == Zero || y == Zero) Some(Zero)
              else if (x == MinusOne) Some(y)
              else if (y == MinusOne) Some(x)
              else None

            case BitOr() if isIntegral(typ) =>
              if (x == MinusOne || y == MinusOne) Some(MinusOne)
              else if (x == Zero) Some(y)
              else if (y == Zero) Some(x)
              else None

            case BitXOr() if isIntegral(typ) =>
              if (x == y) Some(Zero)
              else if (x == Zero) Some(y)
              else if (y == Zero) Some(x)
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

      case x @ If(NA(_), _, _) =>
        Some(NA(x.typ))

      case Coalesce(values) if isDefinitelyDefined(values.head) =>
        Some(values.head)

      case Coalesce(values) if values.zipWithIndex.exists { case (ir, i) =>
            isDefinitelyDefined(ir) && i != values.size - 1
          } =>
        val idx = values.indexWhere(isDefinitelyDefined)
        Some(Coalesce(values.take(idx + 1)))

      case Coalesce(values) if values.size == 1 =>
        Some(values.head)

      case x @ StreamMap(NA(_), _, _) =>
        Some(NA(x.typ))

      case StreamZip(as, names, body, _, _) if as.length == 1 =>
        Some(StreamMap(as.head, names.head, body))

      case StreamMap(StreamZip(as, names, zipBody, b, errorID), name, mapBody) =>
        Some(StreamZip(as, names, Let(FastSeq(name -> zipBody), mapBody), b, errorID))

      case StreamMap(StreamFlatMap(child, flatMapName, flatMapBody), mapName, mapBody) =>
        Some(StreamFlatMap(child, flatMapName, StreamMap(flatMapBody, mapName, mapBody)))

      case x @ (
            StreamFlatMap(NA(_), _, _) |
            StreamFilter(NA(_), _, _) |
            StreamFold(NA(_), _, _, _, _)
          ) =>
        Some(NA(x.typ))

      case IsNA(NA(_)) =>
        Some(True())

      case IsNA(x) if isDefinitelyDefined(x) =>
        Some(False())

      case If(True(), cnsq, _) =>
        Some(cnsq)

      case If(False(), _, altr) =>
        Some(altr)

      case If(c, cnsq, altr) if cnsq == altr && cnsq.typ != TVoid =>
        if (isDefinitelyDefined(c)) Some(cnsq)
        else Some(If(IsNA(c), NA(cnsq.typ), cnsq))

      case If(IsNA(a), NA(_), b) if a == b =>
        Some(b)

      case If(ApplyUnaryPrimOp(Bang, c), cnsq, altr) =>
        Some(If(c, altr, cnsq))

      case If(c1, If(c2, cnsq2, _), altr1) if c1 == c2 =>
        Some(If(c1, cnsq2, altr1))

      case If(c1, cnsq1, If(c2, _, altr2)) if c1 == c2 =>
        Some(If(c1, cnsq1, altr2))

      case Switch(I32(x), default, cases) =>
        Some(if (x >= 0 && x < cases.length) cases(x) else default)

      case Switch(x, default, IndexedSeq()) if isDefinitelyDefined(x) =>
        Some(default)

      case Cast(x, t) if x.typ == t =>
        Some(x)

      case Cast(Cast(x, t2), t) if x.typ == t && Casts.get(t, t2).isLossless =>
        Some(x)

      case CastRename(x, t) if x.typ == t =>
        Some(x)

      case CastRename(CastRename(x, _), t) =>
        Some(CastRename(x, t))

      case ApplyIR("indexArray", _, Seq(a, i @ I32(v)), _, errorID) if v >= 0 =>
        Some(ArrayRef(a, i, errorID))

      case ApplyIR("contains", _, Seq(CastToArray(x), element), _, _) if x.typ.isInstanceOf[TSet] =>
        Some(invoke("contains", TBoolean, x, element))

      case ApplyIR("contains", _, Seq(Literal(t, v), element), _, _) if t.isInstanceOf[TArray] =>
        Some(
          invoke(
            "contains",
            TBoolean,
            Literal(TSet(t.asInstanceOf[TArray].elementType), v.asInstanceOf[IndexedSeq[_]].toSet),
            element,
          )
        )

      case ApplyIR("contains", _, Seq(ToSet(x), element), _, _) if x.typ.isInstanceOf[TArray] =>
        Some(invoke("contains", TBoolean, x, element))

      case x: ApplyIR if x.inline || x.body.size < 10 =>
        Some(x.explicitNode)

      case ArrayLen(MakeArray(args, _)) =>
        Some(I32(args.length))

      case StreamLen(MakeStream(args, _, _)) =>
        Some(I32(args.length))

      case StreamLen(Block(bindings, body)) =>
        Some(Block(bindings, StreamLen(body)))

      case StreamLen(StreamMap(s, _, _)) =>
        Some(StreamLen(s))

      case StreamLen(StreamFlatMap(a, name, body)) =>
        Some(streamSumIR(StreamMap(a, name, StreamLen(body))))

      case StreamLen(StreamGrouped(a, groupSize)) =>
        Some(bindIR(groupSize)(groupSizeRef =>
          (StreamLen(a) + groupSizeRef - 1) floorDiv groupSizeRef
        ))

      case ArrayLen(ToArray(s)) if s.typ.isInstanceOf[TStream] =>
        Some(StreamLen(s))

      case ArrayLen(StreamFlatMap(a, _, MakeArray(args, _))) =>
        Some(ApplyBinaryPrimOp(Multiply(), I32(args.length), ArrayLen(a)))

      case ArrayLen(ArraySort(a, _, _, _)) =>
        Some(ArrayLen(ToArray(a)))

      case ArrayLen(ToArray(MakeStream(args, _, _))) =>
        Some(I32(args.length))

      case ArraySlice(ToArray(s), I32(0), Some(x @ I32(i)), I32(1), _) if i >= 0 =>
        Some(ToArray(StreamTake(s, x)))

      case ArraySlice(z @ ToArray(s), x @ I32(i), Some(I32(j)), I32(1), _) if i > 0 && j > 0 =>
        Some(
          if (j > i) ToArray(StreamTake(StreamDrop(s, x), I32(j - i)))
          else new MakeArray(FastSeq(), z.typ.asInstanceOf[TArray])
        )

      case ArraySlice(ToArray(s), x @ I32(i), None, I32(1), _) if i >= 0 =>
        Some(ToArray(StreamDrop(s, x)))

      case ArrayRef(MakeArray(args, _), I32(i), _) if i >= 0 && i < args.length =>
        Some(args(i))

      case StreamFilter(a, _, True()) =>
        Some(a)

      case StreamFor(_, _, Void()) =>
        Some(Void())

      // FIXME: Unqualify when StreamFold supports folding over stream of streams
      case StreamFold(StreamMap(a, n1, b), zero, accumName, valueName, body)
          if a.typ.asInstanceOf[TStream].elementType.isRealizable =>
        Some(StreamFold(a, zero, accumName, n1, Let(FastSeq(valueName -> b), body)))

      case StreamFlatMap(StreamMap(a, n1, b1), n2, b2) =>
        Some(StreamFlatMap(a, n1, Let(FastSeq(n2 -> b1), b2)))

      case StreamMap(a, elt, r: Ref) if r.name == elt =>
        Some(a)

      case StreamMap(StreamMap(a, n1, b1), n2, b2) =>
        Some(StreamMap(a, n1, Let(FastSeq(n2 -> b1), b2)))

      case StreamFilter(ArraySort(a, left, right, lessThan), name, cond) =>
        Some(ArraySort(StreamFilter(a, name, cond), left, right, lessThan))

      case StreamFilter(
            ToStream(ArraySort(a, left, right, lessThan), requiresMemoryManagementPerElement),
            name,
            cond,
          ) =>
        Some(
          ToStream(
            ArraySort(StreamFilter(a, name, cond), left, right, lessThan),
            requiresMemoryManagementPerElement,
          )
        )

      case CastToArray(x) if x.typ.isInstanceOf[TArray] =>
        Some(x)

      case ToArray(ToStream(a, _)) if a.typ.isInstanceOf[TArray] =>
        Some(a)

      case ToArray(ToStream(a, _)) if a.typ.isInstanceOf[TSet] || a.typ.isInstanceOf[TDict] =>
        Some(CastToArray(a))

      case ToStream(ToArray(s), false) if s.typ.isInstanceOf[TStream] =>
        Some(s)

      case ToStream(Block(bindings, ToArray(x)), false) if x.typ.isInstanceOf[TStream] =>
        Some(Block(bindings, x))

      case MakeNDArray(ToArray(someStream), shape, rowMajor, errorId) =>
        Some(MakeNDArray(someStream, shape, rowMajor, errorId))

      case MakeNDArray(ToStream(someArray, _), shape, rowMajor, errorId) =>
        Some(MakeNDArray(someArray, shape, rowMajor, errorId))

      case NDArrayShape(MakeNDArray(data, shape, _, _)) =>
        Some(If(IsNA(data), NA(shape.typ), shape))

      case NDArrayShape(NDArrayMap(nd, _, _)) =>
        Some(NDArrayShape(nd))

      case NDArrayMap(NDArrayMap(child, innerName, innerBody), outerName, outerBody) =>
        Some(NDArrayMap(child, innerName, Let(FastSeq(outerName -> innerBody), outerBody)))

      case GetField(MakeStruct(fields), name) =>
        val (_, x) = fields.find { case (n, _) => n == name }.get
        Some(x)

      case GetField(InsertFields(old, fields, _), name) =>
        fields.find { case (n, _) => n == name } match {
          case Some((_, x)) => Some(x)
          case None => Some(GetField(old, name))
        }

      case GetField(SelectFields(old, _), name) =>
        Some(GetField(old, name))

      case outer @ InsertFields(InsertFields(base, fields1, fieldOrder1), fields2, fieldOrder2) =>
        val fields2Set = fields2.map(_._1).toSet
        val newFields = fields1.filter { case (name, _) => !fields2Set.contains(name) } ++ fields2
        (fieldOrder1, fieldOrder2) match {
          case (Some(fo1), None) =>
            val fields1Set = fo1.toSet
            val fieldOrder = fo1 ++ fields2.map(_._1).filter(!fields1Set.contains(_))
            Some(InsertFields(base, newFields, Some(fieldOrder)))
          case (_, Some(_)) =>
            Some(InsertFields(base, newFields, fieldOrder2))
          case _ =>
            /* In this case, it's important to make a field order that reflects the original
             * insertion order */
            val resultFieldOrder = outer.typ.fieldNames
            Some(InsertFields(base, newFields, Some(resultFieldOrder)))
        }
      case InsertFields(MakeStruct(fields1), fields2, fieldOrder) =>
        val fields1Map = fields1.toMap
        val fields2Map = fields2.toMap

        fieldOrder match {
          case Some(fo) =>
            Some(MakeStruct(fo.map(f => f -> fields2Map.getOrElse(f, fields1Map(f)))))
          case None =>
            val finalFields = fields1.map { case (name, fieldIR) =>
              name -> fields2Map.getOrElse(name, fieldIR)
            } ++
              fields2.filter { case (name, _) => !fields1Map.contains(name) }
            Some(MakeStruct(finalFields))
        }

      case InsertFields(struct, Seq(), None) =>
        Some(struct)

      case InsertFields(SelectFields(old, _), Seq(), Some(insertFieldOrder)) =>
        Some(SelectFields(old, insertFieldOrder))

      case Block(Seq(), body) =>
        Some(body)

      case Block(xs, Block(ys, body)) =>
        Some(Block(xs ++ ys, body))

      // assumes `NormalizeNames` has been run before this.
      case Block(Block.Nested(i, bindings), body) =>
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
                               case Binding(name, value, Scope.EVAL) => Binding(name, value, scope)
                               case _ => fatal("Simplify: found nested Agg bindings")
                             })
            newBindings += Binding(name, ir.body, scope)
          case binding => newBindings += binding
        }

        Some(Block(newBindings.result(), body))

      case Block(
            Block.Insert(
              before,
              Binding(name, x @ InsertFields(old, newFields, _), Scope.EVAL),
              after,
            ),
            body,
          )
          if x.typ.size < 500 && {
            val r = Ref(name, x.typ)
            val nfSet = newFields.map(_._1).toSet

            def allRefsCanBePassedThrough(ir1: IR): Boolean = ir1 match {
              case GetField(`r`, _) => true
              case InsertFields(`r`, inserted, _) =>
                inserted.forall { case (_, toInsert) => allRefsCanBePassedThrough(toInsert) }
              case SelectFields(`r`, fds) => fds.forall(f => !nfSet.contains(f))
              case `r` => false // if the binding is referenced in any other context, don't rewrite
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

            allRefsCanBePassedThrough(Block(after.toFastSeq, body))
          } =>
        val fieldNames = newFields.map(_._1).toArray
        val newFieldMap = newFields.toMap
        val newFieldRefs = newFieldMap.map { case (k, ir) =>
          (k, Ref(freshName(), ir.typ))
        } // cannot be mapValues, or genUID() gets run for every usage!

        def copiedNewFieldRefs(): IndexedSeq[(String, IR)] =
          fieldNames.map(name => (name, newFieldRefs(name).deepCopy())).toFastSeq

        def rewrite(ir1: IR): IR = ir1 match {
          case GetField(Ref(`name`, _), fd) => newFieldRefs.get(fd) match {
              case Some(r) => r.deepCopy()
              case None => GetField(Ref(name, old.typ), fd)
            }
          case ins @ InsertFields(Ref(`name`, _), fields, _) =>
            val newFieldSet = fields.map(_._1).toSet
            InsertFields(
              Ref(name, old.typ),
              copiedNewFieldRefs().filter { case (name, _) => !newFieldSet.contains(name) }
                ++ fields.map { case (name, ir) => (name, rewrite(ir)) },
              Some(ins.typ.fieldNames.toFastSeq),
            )

          case SelectFields(Ref(`name`, _), fds) =>
            SelectFields(
              InsertFields(
                Ref(name, old.typ),
                copiedNewFieldRefs(),
                Some(x.typ.fieldNames.toFastSeq),
              ),
              fds,
            )
          case ta: TableAggregate => ta
          case ma: MatrixAggregate => ma
          case _ => ir1.mapChildrenWithIndex {
              case (child: IR, idx) => if (Binds(ir1, name, idx)) child else rewrite(child)
              case (child, _) => child
            }
        }

        Some(
          Block(
            before.toFastSeq ++ fieldNames.map(f =>
              Binding(newFieldRefs(f).name, newFieldMap(f))
            ) ++ FastSeq(
              Binding(name, old)
            ),
            rewrite(Block(after.toFastSeq, body)),
          )
        )

      case SelectFields(old, fields) if tcoerce[TStruct](old.typ).fieldNames sameElements fields =>
        Some(old)

      case SelectFields(SelectFields(old, _), fields) =>
        Some(SelectFields(old, fields))

      case SelectFields(MakeStruct(fields), fieldNames) =>
        val makeStructFields = fields.toMap
        Some(MakeStruct(fieldNames.map(f => f -> makeStructFields(f))))

      case x @ SelectFields(InsertFields(struct, insertFields, _), selectFields) =>
        val selectSet = selectFields.toSet
        val insertFields2 = insertFields.filter { case (fName, _) => selectSet.contains(fName) }
        val structSet = struct.typ.asInstanceOf[TStruct].fieldNames.toSet
        val selectFields2 = selectFields.filter(structSet.contains)
        val x2 = InsertFields(
          SelectFields(struct, selectFields2),
          insertFields2,
          Some(selectFields.toFastSeq),
        )
        assert(x2.typ == x.typ)
        Some(x2)

      case x @ InsertFields(SelectFields(struct, selectFields), insertFields, _) if
            insertFields.exists { case (name, f) => f == GetField(struct, name) } =>
        val fields = x.typ.fieldNames
        val insertNames = insertFields.map(_._1).toSet
        val (oldFields, newFields) =
          insertFields.partition { case (name, f) => f == GetField(struct, name) }
        val preservedFields =
          selectFields.filter(f => !insertNames.contains(f)) ++ oldFields.map(_._1)
        Some(InsertFields(SelectFields(struct, preservedFields), newFields, Some(fields.toFastSeq)))

      case MakeStructOfGetField(o, newNames) =>
        val select = SelectFields(o, newNames.map(_._1))
        Some(CastRename(select, select.typ.rename(newNames.toMap)))

      case GetTupleElement(MakeTuple(xs), idx) =>
        Some(xs.find(_._1 == idx).get._2)

      case TableCount(MatrixColsTable(ColumnCount(nCols))) =>
        Some(I64(nCols.toLong))

      case TableCount(PartitionCounts(counts)) =>
        Some(I64(counts.sum))

      case TableCount(CastMatrixToTable(child, _, _)) =>
        Some(TableCount(MatrixRowsTable(child)))

      case TableCount(TableMapGlobals(child, _)) =>
        Some(TableCount(child))

      case TableCount(TableMapRows(child, _)) =>
        Some(TableCount(child))

      case TableCount(TableRepartition(child, _, _)) =>
        Some(TableCount(child))

      case TableCount(TableUnion(children)) =>
        Some(children.map(TableCount(_): IR).treeReduce(ApplyBinaryPrimOp(Add(), _, _)))

      case TableCount(TableKeyBy(child, _, _)) =>
        Some(TableCount(child))

      case TableCount(TableOrderBy(child, _)) =>
        Some(TableCount(child))

      case TableCount(TableLeftJoinRightDistinct(child, _, _)) =>
        Some(TableCount(child))

      case TableCount(TableIntervalJoin(child, _, _, _)) =>
        Some(TableCount(child))

      case TableCount(TableRange(n, _)) =>
        Some(I64(n.toLong))

      case TableCount(TableParallelize(rowsAndGlobal, _)) =>
        Some(Cast(ArrayLen(GetField(rowsAndGlobal, "rows")), TInt64))

      case TableCount(TableRename(child, _, _)) =>
        Some(TableCount(child))

      case TableCount(TableAggregateByKey(child, _)) =>
        Some(TableCount(TableDistinct(child)))

      case TableCount(TableExplode(child, path)) =>
        Some(
          TableAggregate(
            child,
            ApplyAggOp(
              FastSeq(),
              FastSeq(ArrayLen(CastToArray(path.foldLeft[IR](Ref(
                TableIR.rowName,
                child.typ.rowType,
              )) {
                case (comb, s) => GetField(comb, s)
              })).toL),
              Sum(),
            ),
          )
        )

      case MatrixCount(child @ PartitionCountsOrColumnCount((maybeParts, maybeCols))) =>
        val rowCount = maybeParts match {
          case Some(pc) => I64(pc.sum)
          case None => TableCount(MatrixRowsTable(child))
        }
        val colCount = maybeCols match {
          case Some(cc) => I32(cc)
          case None => TableCount(MatrixColsTable(child)).toI
        }
        Some(MakeTuple.ordered(FastSeq(rowCount, colCount)))

      case MatrixCount(MatrixMapRows(child, _)) =>
        Some(MatrixCount(child))

      case MatrixCount(MatrixMapCols(child, _, _)) =>
        Some(MatrixCount(child))

      case MatrixCount(MatrixMapEntries(child, _)) =>
        Some(MatrixCount(child))

      case MatrixCount(MatrixFilterEntries(child, _)) =>
        Some(MatrixCount(child))

      case MatrixCount(MatrixAnnotateColsTable(child, _, _)) =>
        Some(MatrixCount(child))

      case MatrixCount(MatrixAnnotateRowsTable(child, _, _, _)) =>
        Some(MatrixCount(child))

      case MatrixCount(MatrixRepartition(child, _, _)) =>
        Some(MatrixCount(child))
      case MatrixCount(MatrixRename(child, _, _, _, _)) =>
        Some(MatrixCount(child))
      case TableCount(TableRead(_, false, r: MatrixBGENReader))
          if r.params.includedVariants.isEmpty =>
        Some(I64(r.nVariants))

      // TableGetGlobals should simplify very aggressively
      case TableGetGlobals(child) if child.typ.globalType == TStruct.empty =>
        Some(MakeStruct(FastSeq()))

      case TableGetGlobals(TableKeyBy(child, _, _)) =>
        Some(TableGetGlobals(child))

      case TableGetGlobals(TableFilter(child, _)) =>
        Some(TableGetGlobals(child))

      case TableGetGlobals(TableHead(child, _)) =>
        Some(TableGetGlobals(child))

      case TableGetGlobals(TableRepartition(child, _, _)) =>
        Some(TableGetGlobals(child))

      case TableGetGlobals(TableJoin(child1, child2, _, _)) =>
        Some(
          bindIRs(TableGetGlobals(child1), TableGetGlobals(child2)) { case Seq(g1, g2) =>
            MakeStruct(
              g1.typ.asInstanceOf[TStruct].fields.map(f => f.name -> GetField(g1, f.name)) ++
                g2.typ.asInstanceOf[TStruct].fields.map(f => f.name -> GetField(g2, f.name))
            )
          }
        )

      case TableGetGlobals(TableMultiWayZipJoin(children, _, globalName)) =>
        Some(
          MakeStruct(FastSeq(globalName -> MakeArray(
            children.map(TableGetGlobals),
            TArray(children.head.typ.globalType),
          )))
        )

      case TableGetGlobals(TableLeftJoinRightDistinct(child, _, _)) =>
        Some(TableGetGlobals(child))

      case TableGetGlobals(TableMapRows(child, _)) =>
        Some(TableGetGlobals(child))

      case TableGetGlobals(TableMapGlobals(child, newGlobals)) =>
        Some(
          bindIR(TableGetGlobals(child)) { ref =>
            Subst(newGlobals, BindingEnv(Env.empty[IR].bind(TableIR.globalName, ref)))
          }
        )

      case TableGetGlobals(TableExplode(child, _)) =>
        Some(TableGetGlobals(child))

      case TableGetGlobals(TableUnion(children)) =>
        Some(TableGetGlobals(children.head))

      case TableGetGlobals(TableDistinct(child)) =>
        Some(TableGetGlobals(child))

      case TableGetGlobals(TableAggregateByKey(child, _)) =>
        Some(TableGetGlobals(child))

      case TableGetGlobals(TableKeyByAndAggregate(child, _, _, _, _)) =>
        Some(TableGetGlobals(child))

      case TableGetGlobals(TableOrderBy(child, _)) =>
        Some(TableGetGlobals(child))

      case TableGetGlobals(TableRename(child, _, globalMap)) =>
        if (globalMap.isEmpty)
          Some(TableGetGlobals(child))
        else
          Some(bindIR(TableGetGlobals(child)) { ref =>
            MakeStruct(child.typ.globalType.fieldNames.map { f =>
              globalMap.getOrElse(f, f) -> GetField(ref, f)
            })
          })

      case TableCollect(TableParallelize(x, _)) =>
        Some(x)

      case x @ TableCollect(TableOrderBy(child, sortFields))
          if sortFields.forall(_.sortOrder == Ascending)
            && !child.typ.key.startsWith(sortFields.map(_.field)) =>
        val uid = freshName()
        val uid2 = freshName()
        val left = freshName()
        val right = freshName()
        val uid3 = freshName()

        val kvElement = MakeStruct(FastSeq(
          ("key", SelectFields(Ref(uid2, child.typ.rowType), sortFields.map(_.field))),
          ("value", Ref(uid2, child.typ.rowType)),
        ))
        val sorted = ArraySort(
          StreamMap(
            ToStream(GetField(Ref(uid, x.typ), "rows")),
            uid2,
            kvElement,
          ),
          left,
          right,
          ApplyComparisonOp(
            LT,
            GetField(Ref(left, kvElement.typ), "key"),
            GetField(Ref(right, kvElement.typ), "key"),
          ),
        )
        val block = Let(
          FastSeq(uid -> TableCollect(TableKeyBy(child, FastSeq()))),
          MakeStruct(FastSeq(
            (
              "rows",
              ToArray(StreamMap(
                ToStream(sorted),
                uid3,
                GetField(Ref(uid3, sorted.typ.asInstanceOf[TArray].elementType), "value"),
              )),
            ),
            ("global", GetField(Ref(uid, x.typ), "global")),
          )),
        )
        Some(block)

      case ArrayLen(GetField(TableCollect(child), "rows")) =>
        Some(Cast(TableCount(child), TInt32))
      case GetField(TableCollect(child), "global") =>
        Some(TableGetGlobals(child))

      case TableAggregate(child, query)
          if child.typ.key.nonEmpty && !ContainsNonCommutativeAgg(query) =>
        Some(TableAggregate(TableKeyBy(child, FastSeq(), false), query))

      case TableAggregate(TableOrderBy(child, _), query) if !ContainsNonCommutativeAgg(query) =>
        if (child.typ.key.isEmpty)
          Some(TableAggregate(child, query))
        else
          Some(TableAggregate(TableKeyBy(child, FastSeq(), false), query))

      case TableAggregate(TableMapRows(child, newRow), query) if !ContainsScan(newRow) =>
        val uid = freshName()
        val agg = TableAggregate(
          child,
          AggLet(
            uid,
            newRow,
            Subst(query, BindingEnv(agg = Some(Env(TableIR.rowName -> Ref(uid, newRow.typ))))),
            isScan = false,
          ),
        )
        Some(agg)

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

      case ApplyIR("annotate", _, Seq(s, MakeStruct(fields)), _, _) =>
        Some(InsertFields(s, fields))

      // simplify Boolean equality
      case ApplyComparisonOp(EQ, expr, True()) =>
        Some(expr)

      case ApplyComparisonOp(EQ, True(), expr) =>
        Some(expr)

      case ApplyComparisonOp(EQ, expr, False()) =>
        Some(ApplyUnaryPrimOp(Bang, expr))

      case ApplyComparisonOp(EQ, False(), expr) =>
        Some(ApplyUnaryPrimOp(Bang, expr))

      case ApplyUnaryPrimOp(Bang, ApplyComparisonOp(op, l, r)) =>
        Some(ApplyComparisonOp(ComparisonOp.negate(op.asInstanceOf[ComparisonOp[Boolean]]), l, r))

      case StreamAgg(_, _, query) if !ContainsAgg(query) =>
        Some(query)

      case StreamAggScan(a, name, query) if !ContainsScan(query) =>
        Some(StreamMap(a, name, query))

      case BlockMatrixToValueApply(
            ValueToBlockMatrix(child, IndexedSeq(_, ncols), _),
            functions.GetElement(Seq(i, j)),
          ) => child.typ match {
          case TArray(_) => Some(ArrayRef(child, I32((i * ncols + j).toInt)))
          case TNDArray(_, _) => Some(NDArrayRef(child, IndexedSeq(i, j), ErrorIDs.NO_ERROR))
          case TFloat64 => Some(child)
        }
      case LiftMeOut(child) if IsConstant(child) =>
        Some(child)

      case _ =>
        None
    }

  private[this] def tableRules(ctx: ExecuteContext, tir: TableIR): Option[TableIR] =
    tir match {
      case TableRename(child, m1, m2) if m1.isTrivial && m2.isTrivial =>
        Some(child)

      // TODO: Write more rules like this to bubble 'TableRename' nodes towards the root.
      case t @ TableRename(TableKeyBy(child, keys, isSorted), rowMap, globalMap) =>
        Some(TableKeyBy(TableRename(child, rowMap, globalMap), keys.map(t.rowF), isSorted))

      case TableFilter(t, True()) =>
        Some(t)

      case TableFilter(TableRead(typ, _, tr), False() | NA(_)) =>
        Some(TableRead(typ, dropRows = true, tr))

      case TableFilter(TableFilter(t, p1), p2) =>
        Some(TableFilter(
          t,
          ApplySpecial("land", Array.empty[Type], Array(p1, p2), TBoolean, ErrorIDs.NO_ERROR),
        ))

      case TableFilter(TableKeyBy(child, key, isSorted), p) =>
        Some(TableKeyBy(TableFilter(child, p), key, isSorted))

      case TableFilter(TableRepartition(child, n, strategy), p) =>
        Some(TableRepartition(TableFilter(child, p), n, strategy))

      case TableOrderBy(TableKeyBy(child, _, false), sortFields) =>
        Some(TableOrderBy(child, sortFields))

      case TableFilter(TableOrderBy(child, sortFields), pred) =>
        Some(TableOrderBy(TableFilter(child, pred), sortFields))

      case TableFilter(TableParallelize(rowsAndGlobal, nPartitions), pred) =>
        val newRowsAndGlobal = rowsAndGlobal match {
          case MakeStruct(Seq(("rows", rows), ("global", globalVal))) =>
            Let(
              FastSeq(TableIR.globalName -> globalVal),
              MakeStruct(FastSeq(
                ("rows", ToArray(StreamFilter(ToStream(rows), TableIR.rowName, pred))),
                ("global", Ref(TableIR.globalName, globalVal.typ)),
              )),
            )
          case _ =>
            val uid = freshName()
            Let(
              FastSeq(
                uid -> rowsAndGlobal,
                TableIR.globalName -> GetField(Ref(uid, rowsAndGlobal.typ), "global"),
              ),
              MakeStruct(FastSeq(
                "rows" -> ToArray(StreamFilter(
                  ToStream(GetField(Ref(uid, rowsAndGlobal.typ), "rows")),
                  TableIR.rowName,
                  pred,
                )),
                "global" -> Ref(
                  TableIR.globalName,
                  rowsAndGlobal.typ.asInstanceOf[TStruct].fieldType("global"),
                ),
              )),
            )
        }
        Some(TableParallelize(newRowsAndGlobal, nPartitions))

      case TableKeyBy(TableOrderBy(child, _), keys, false) =>
        Some(TableKeyBy(child, keys, false))

      case TableKeyBy(TableKeyBy(child, _, _), keys, false) =>
        Some(TableKeyBy(child, keys, false))

      case TableKeyBy(TableKeyBy(child, _, true), keys, true) =>
        Some(TableKeyBy(child, keys, true))

      case TableKeyBy(child, key, _) if key == child.typ.key =>
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

      case TableMapRows(TableMapRows(child, newRow1), newRow2) if !ContainsScan(newRow2) =>
        Some(TableMapRows(child, Let(FastSeq(TableIR.rowName -> newRow1), newRow2)))

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
        val row = Ref(TableIR.rowName, child.typ.rowType)
        val keyStruct = MakeStruct(sortFields.map(f => f.field -> GetField(row, f.field)))
        val te =
          TableExplode(
            TableKeyByAndAggregate(
              child,
              MakeStruct(FastSeq(
                "row" -> ApplyAggOp(
                  FastSeq(I32(n.toInt)),
                  Array(row, keyStruct),
                  TakeBy(),
                )
              )),
              MakeStruct(FastSeq()), // aggregate to one row
              Some(1),
              10,
            ),
            FastSeq("row"),
          )
        Some(TableMapRows(te, GetField(Ref(TableIR.rowName, te.typ.rowType), "row")))

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

      case TableAggregateByKey(x @ TableKeyBy(child, keys, false), expr)
          if !x.definitelyDoesNotShuffle =>
        Some(TableKeyByAndAggregate(
          child,
          expr,
          MakeStruct(keys.map(k => k -> GetField(Ref(TableIR.rowName, child.typ.rowType), k))),
          bufferSize = ctx.getFlag("grouped_aggregate_buffer_size").toInt,
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
        val i1 = Interval.union(_i1.toArray[Interval], ord)
        val i2 = Interval.union(_i2.toArray[Interval], ord)
        val intervals = if (keep1)
          // keep means intersect intervals
          Interval.intersection(i1, i2, ord)
        else
          // remove means union intervals
          Interval.union(i1 ++ i2, ord)
        Some(TableFilterIntervals(child, intervals.toFastSeq, keep1))

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
        val getElement = BlockMatrixToValueApply(scalarBM, functions.GetElement(IndexedSeq(0, 0)))
        val needsDense = sparsityStrategy == NeedsDense || sparsityStrategy.exists(
          leftBlock = true,
          rightBlock = false,
        )
        val maybeDense = if (needsDense) BlockMatrixDensify(right) else right
        Some(
          BlockMatrixMap(
            maybeDense,
            rightName,
            Subst(f, BindingEnv.eval(leftName -> getElement)),
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
        val getElement = BlockMatrixToValueApply(scalarBM, functions.GetElement(IndexedSeq(0, 0)))
        val needsDense = sparsityStrategy == NeedsDense || sparsityStrategy.exists(
          leftBlock = false,
          rightBlock = true,
        )
        val maybeDense = if (needsDense) BlockMatrixDensify(left) else left
        Some(
          BlockMatrixMap(
            maybeDense,
            leftName,
            Subst(f, BindingEnv.eval(rightName -> getElement)),
            needsDense,
          )
        )
      case BlockMatrixMap(matrix, name, Ref(x, _), _) if name == x =>
        Some(matrix)
      case BlockMatrixMap(matrix, name, ir, _)
          if IsConstant(ir) || (ir.isInstanceOf[Ref] && ir.asInstanceOf[Ref].name != name) =>
        val typ = matrix.typ
        Some(BlockMatrixBroadcast(
          ValueToBlockMatrix(ir, FastSeq(1, 1), typ.blockSize),
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

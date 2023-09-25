package is.hail.expr.ir

import is.hail.backend.ExecuteContext
import is.hail.io.bgen.MatrixBGENReader
import is.hail.rvd.PartitionBoundOrdering
import is.hail.types.tcoerce
import is.hail.types.virtual._
import is.hail.utils._

object Simplify {

  /** Transform 'ir' using simplification rules until none apply.
    */
  def apply(ctx: ExecuteContext, ir: BaseIR): BaseIR = ir match {
    case ir: IR => simplifyValue(ctx)(ir)
    case tir: TableIR => simplifyTable(ctx)(tir)
    case mir: MatrixIR => simplifyMatrix(ctx)(mir)
    case bmir: BlockMatrixIR => simplifyBlockMatrix(ctx)(bmir)
  }

  private[this] def visitNode[T <: BaseIR](
    visitChildren: BaseIR => BaseIR,
    transform: T => Option[T],
    post: => (T => T)
  )(t: T): T = {
    val t1 = t.mapChildren(visitChildren).asInstanceOf[T]
    transform(t1).map(post).getOrElse(t1)
  }

  private[this] def simplifyValue(ctx: ExecuteContext): IR => IR =
    visitNode(
      Simplify(ctx, _),
      rewriteValueNode,
      simplifyValue(ctx))

  private[this] def simplifyTable(ctx: ExecuteContext)(tir: TableIR): TableIR =
    visitNode(
      Simplify(ctx, _),
      rewriteTableNode(ctx),
      simplifyTable(ctx)
    )(tir)

  private[this] def simplifyMatrix(ctx: ExecuteContext)(mir: MatrixIR): MatrixIR =
    visitNode(
      Simplify(ctx, _),
      rewriteMatrixNode(),
      simplifyMatrix(ctx)
    )(mir)

  private[this] def simplifyBlockMatrix(ctx: ExecuteContext)(bmir: BlockMatrixIR): BlockMatrixIR = {
    visitNode(
      Simplify(ctx, _),
      rewriteBlockMatrixNode,
      simplifyBlockMatrix(ctx)
    )(bmir)
  }

  private[this] def rewriteValueNode(ir: IR): Option[IR] =
    valueRules.lift(ir).orElse(numericRules(ir))

  private[this] def rewriteTableNode(ctx: ExecuteContext)(tir: TableIR): Option[TableIR] =
    tableRules(ctx).lift(tir)

  private[this] def rewriteMatrixNode()(mir: MatrixIR): Option[MatrixIR] =
    matrixRules().lift(mir)

  private[this] def rewriteBlockMatrixNode: BlockMatrixIR => Option[BlockMatrixIR] = blockMatrixRules.lift

  /** Returns true if 'x' propagates missingness, meaning if any child of 'x'
    * evaluates to missing, then 'x' will evaluate to missing.
    */
  private[this] def isStrict(x: IR): Boolean = {
    x match {
      case _: Apply |
           _: ApplySeeded |
           _: ApplyUnaryPrimOp |
           _: ApplyBinaryPrimOp |
           _: ArrayRef |
           _: ArrayLen |
           _: GetField |
           _: GetTupleElement => true
      case ApplyComparisonOp(op, _, _) => op.strict
      case _ => false
    }
  }

  /**
    * Returns true if any strict child of 'x' is NA.
    * A child is strict if 'x' evaluates to missing whenever the child does.
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

  /** Returns true if 'x' will never evaluate to missing.
    */
  private[this] def isDefinitelyDefined(x: IR): Boolean = {
    x match {
      case _: MakeArray |
           _: MakeStruct |
           _: MakeTuple |
           _: IsNA |
           ApplyComparisonOp(EQWithNA(_, _), _, _) |
           ApplyComparisonOp(NEQWithNA(_, _), _, _) |
           _: I32 | _: I64 | _: F32 | _: F64 | True() | False() => true
      case _ => false
    }
  }

  private def numericRules: IR => Option[IR] = {

    def integralBinaryIdentities(pure: Int => IR) = (ir: IR) => ir match {
      case ApplyBinaryPrimOp(op, x, y) if ir.typ.isInstanceOf[TIntegral] =>
        op match {
          case Add() =>
            if (x == y) Some(ApplyBinaryPrimOp(Multiply(), pure(2), x))
            else None

          case Subtract() =>
            if (x == y) Some(pure(0))
            else None

          case Multiply() =>
            if (x == pure(0) || y == pure(0)) Some(pure(0))
            else None

          case RoundToNegInfDivide() =>
            if (x == y) Some(pure(1))
            else if (x == pure(0)) Some(pure(0))
            else if (y == pure(0)) Some(Die("division by zero", ir.typ))
            else None

          case _: LeftShift | _:RightShift | _: LogicalRightShift  =>
            if (x == pure(0)) Some(pure(0))
            else if (y == I32(0)) Some(x)
            else None

          case BitAnd() =>
            if (x == pure(0) || y == pure(0)) Some(pure(0))
            else if (x == pure(-1)) Some(y)
            else if (y == pure(-1)) Some(x)
            else None

          case BitOr() =>
            if (x == pure(-1) || y == pure(-1)) Some(pure(-1))
            else if (x == pure(0)) Some(y)
            else if (y == pure(0)) Some(x)
            else None

          case BitXOr() =>
            if (x == y) Some(pure(0))
            else if (x == pure(0)) Some(y)
            else if (y == pure(0)) Some(x)
            else None

          case _ =>
            None
        }
      case _ =>
        None
    }

    def hoistUnaryOp = (ir: IR) => ir match {
      case ApplyUnaryPrimOp(f@(_: Negate | _: BitNot | _: Bang), x) => x match {
        case ApplyUnaryPrimOp(g, y) if g == f => Some(y)
        case _ => None
      }
      case _ => None
    }

    def commonBinaryIdentities(pure: Int => IR) = (ir: IR) => ir match {
      case ApplyBinaryPrimOp(f, x, y) =>
        f match {
          case Add() =>
            if (x == pure(0)) Some(y)
            else if (y == pure(0)) Some(x)
            else None

          case Subtract() =>
            if (x == pure(0)) Some(ApplyUnaryPrimOp(Negate(), y))
            else if (y == pure(0)) Some(x)
            else None

          case Multiply() =>
            if (x == pure(1)) Some(y)
            else if (x == pure(-1)) Some(ApplyUnaryPrimOp(Negate(), y))
            else if (y == pure(1)) Some(x)
            else if (y == pure(-1)) Some(ApplyUnaryPrimOp(Negate(), x))
            else None

          case RoundToNegInfDivide() =>
            if (y == pure(1)) Some(x)
            else if (y == pure(-1)) Some(ApplyUnaryPrimOp(Negate(), x))
            else None

          case _ =>
            None
        }
      case _ =>
        None
    }

    Array(
      hoistUnaryOp,
      (ir: IR) => integralBinaryIdentities(Literal.coerce(ir.typ, _))(ir),
      (ir: IR) => commonBinaryIdentities(Literal.coerce(ir.typ, _))(ir),
    ).reduce((f, g) => ir => f(ir).orElse(g(ir)))
  }

  private[this] def valueRules: PartialFunction[IR, IR] = {
    // propagate NA
    case x: IR if hasMissingStrictChild(x) =>
      NA(x.typ)

    case x@If(NA(_), _, _) => NA(x.typ)

    case Coalesce(values) if isDefinitelyDefined(values.head) => values.head

    case Coalesce(values) if values.zipWithIndex.exists { case (ir, i) => isDefinitelyDefined(ir) && i != values.size - 1 } =>
      val idx = values.indexWhere(isDefinitelyDefined)
      Coalesce(values.take(idx + 1))

    case Coalesce(values) if values.size == 1 => values.head

    case x@StreamMap(NA(_), _, _) => NA(x.typ)

    case StreamZip(as, names, body, _, _) if as.length == 1 => StreamMap(as.head, names.head, body)
    case StreamMap(StreamZip(as, names, zipBody, b, errorID), name, mapBody) => StreamZip(as, names, Let(name, zipBody, mapBody), b, errorID)
    case StreamMap(StreamFlatMap(child, flatMapName, flatMapBody), mapName, mapBody) => StreamFlatMap(child, flatMapName, StreamMap(flatMapBody, mapName, mapBody))

    case x@StreamFlatMap(NA(_), _, _) => NA(x.typ)

    case x@StreamFilter(NA(_), _, _) => NA(x.typ)

    case x@StreamFold(NA(_), _, _, _, _) => NA(x.typ)

    case IsNA(NA(_)) => True()

    case IsNA(x) if isDefinitelyDefined(x) => False()

    case x@If(True(), cnsq, _) => cnsq

    case x@If(False(), _, altr) => altr

    case If(c, cnsq, altr) if cnsq == altr && cnsq.typ != TVoid =>
      if (isDefinitelyDefined(c))
        cnsq
      else
        If(IsNA(c), NA(cnsq.typ), cnsq)

    case If(ApplyUnaryPrimOp(Bang(), c), cnsq, altr) => If(c, altr, cnsq)

    case If(c1, If(c2, cnsq2, _), altr1) if c1 == c2 => If(c1, cnsq2, altr1)

    case If(c1, cnsq1, If(c2, _, altr2)) if c1 == c2 => If(c1, cnsq1, altr2)

    case Cast(x, t) if x.typ == t => x
    case Cast(Cast(x, _), t) if x.typ == t =>x

    case CastRename(x, t) if x.typ == t => x
    case CastRename(CastRename(x, _), t) => CastRename(x, t)

    case ApplyIR("indexArray", _, Seq(a, i@I32(v)), errorID) if v >= 0 =>
      ArrayRef(a, i, errorID)

    case ApplyIR("contains", _, Seq(CastToArray(x), element), _) if x.typ.isInstanceOf[TSet] => invoke("contains", TBoolean, x, element)

    case ApplyIR("contains", _, Seq(Literal(t, v), element), _) if t.isInstanceOf[TArray] =>
      invoke("contains", TBoolean, Literal(TSet(t.asInstanceOf[TArray].elementType), v.asInstanceOf[IndexedSeq[_]].toSet), element)

    case ApplyIR("contains", _, Seq(ToSet(x), element), _) if x.typ.isInstanceOf[TArray] => invoke("contains", TBoolean, x, element)

    case x: ApplyIR if x.inline || x.body.size < 10 => x.explicitNode

    case ArrayLen(MakeArray(args, _)) => I32(args.length)

    case StreamLen(MakeStream(args, _, _)) => I32(args.length)
    case StreamLen(Let(name, value, body)) => Let(name, value, StreamLen(body))
    case StreamLen(StreamMap(s, _, _)) => StreamLen(s)
    case StreamLen(StreamFlatMap(a, name, body)) => streamSumIR(StreamMap(a, name, StreamLen(body)))
    case StreamLen(StreamGrouped(a, groupSize)) => bindIR(groupSize)(groupSizeRef => (StreamLen(a) + groupSizeRef - 1) floorDiv groupSizeRef)

    case ArrayLen(ToArray(s)) if s.typ.isInstanceOf[TStream] => StreamLen(s)
    case ArrayLen(StreamFlatMap(a, _, MakeArray(args, _))) => ApplyBinaryPrimOp(Multiply(), I32(args.length), ArrayLen(a))

    case ArrayLen(ArraySort(a, _, _, _)) => ArrayLen(ToArray(a))

    case ArrayLen(ToArray(MakeStream(args, _, _))) => I32(args.length)

    case ArraySlice(ToArray(s),I32(0), Some(x@I32(i)), I32(1), _) if i >= 0 =>
      ToArray(StreamTake(s, x))

    case ArraySlice(z@ToArray(s), x@I32(i), Some(I32(j)), I32(1), _) if i > 0 && j > 0 => {
      if (j > i) {
        ToArray(StreamTake(StreamDrop(s, x), I32(j-i)))
      } else new MakeArray(FastSeq(), z.typ.asInstanceOf[TArray])
    }

    case ArraySlice(ToArray(s), x@I32(i), None, I32(1), _) if i >= 0 =>
      ToArray(StreamDrop(s, x))

    case ArrayRef(MakeArray(args, _), I32(i), _) if i >= 0 && i < args.length => args(i)

    case StreamFilter(a, _, True()) => a

    case StreamFor(_, _, Begin(Seq())) => Begin(FastSeq())

    // FIXME: Unqualify when StreamFold supports folding over stream of streams
    case StreamFold(StreamMap(a, n1, b), zero, accumName, valueName, body) if a.typ.asInstanceOf[TStream].elementType.isRealizable => StreamFold(a, zero, accumName, n1, Let(valueName, b, body))

    case StreamFlatMap(StreamMap(a, n1, b1), n2, b2) =>
      StreamFlatMap(a, n1, Let(n2, b1, b2))

    case StreamMap(a, elt, r: Ref) if r.name == elt => a

    case StreamMap(StreamMap(a, n1, b1), n2, b2) =>
      StreamMap(a, n1, Let(n2, b1, b2))

    case StreamFilter(ArraySort(a, left, right, lessThan), name, cond) => ArraySort(StreamFilter(a, name, cond), left, right, lessThan)

    case StreamFilter(ToStream(ArraySort(a, left, right, lessThan), requiresMemoryManagementPerElement), name, cond) =>
      ToStream(ArraySort(StreamFilter(a, name, cond), left, right, lessThan), requiresMemoryManagementPerElement)

    case CastToArray(x) if x.typ.isInstanceOf[TArray] => x
    case ToArray(ToStream(a, _)) if a.typ.isInstanceOf[TArray] => a
    case ToArray(ToStream(a, _)) if a.typ.isInstanceOf[TSet] || a.typ.isInstanceOf[TDict] =>
      CastToArray(a)

    case ToStream(ToArray(s), false) if s.typ.isInstanceOf[TStream] => s

    case ToStream(Let(name, value, ToArray(x)), false) if x.typ.isInstanceOf[TStream] =>
      Let(name, value, x)

    case MakeNDArray(ToArray(someStream), shape, rowMajor, errorId) => MakeNDArray(someStream, shape, rowMajor, errorId)
    case MakeNDArray(ToStream(someArray, _), shape, rowMajor, errorId) => MakeNDArray(someArray, shape, rowMajor, errorId)
    case NDArrayShape(MakeNDArray(data, shape, _, _)) => {
      If(IsNA(data), NA(shape.typ), shape)
    }
    case NDArrayShape(NDArrayMap(nd, _, _)) => NDArrayShape(nd)

    case NDArrayMap(NDArrayMap(child, innerName, innerBody), outerName, outerBody) => {
      NDArrayMap(child, innerName, Let(outerName, innerBody, outerBody))
    }

    case GetField(MakeStruct(fields), name) =>
      val (_, x) = fields.find { case (n, _) => n == name }.get
      x

    case GetField(InsertFields(old, fields, _), name) =>
      fields.find { case (n, _) => n == name } match {
        case Some((_, x)) => x
        case None => GetField(old, name)
      }

    case GetField(SelectFields(old, fields), name) => GetField(old, name)

    case outer@InsertFields(InsertFields(base, fields1, fieldOrder1), fields2, fieldOrder2) =>
      val fields2Set = fields2.map(_._1).toSet
      val newFields = fields1.filter { case (name, _) => !fields2Set.contains(name) } ++ fields2
      (fieldOrder1, fieldOrder2) match {
        case (Some(fo1), None) =>
          val fields1Set = fo1.toSet
          val fieldOrder = fo1 ++ fields2.map(_._1).filter(!fields1Set.contains(_))
          InsertFields(base, newFields, Some(fieldOrder))
        case (_, Some(_)) =>
          InsertFields(base, newFields, fieldOrder2)
        case _ =>
          // In this case, it's important to make a field order that reflects the original insertion order
          val resultFieldOrder = outer.typ.fieldNames
          InsertFields(base, newFields, Some(resultFieldOrder))
      }
    case InsertFields(MakeStruct(fields1), fields2, fieldOrder) =>
      val fields1Map = fields1.toMap
      val fields2Map = fields2.toMap

      fieldOrder match {
        case Some(fo) =>
          MakeStruct(fo.map(f => f -> fields2Map.getOrElse(f, fields1Map(f))))
        case None =>
          val finalFields = fields1.map { case (name, fieldIR) => name -> fields2Map.getOrElse(name, fieldIR) } ++
            fields2.filter { case (name, _) => !fields1Map.contains(name) }
          MakeStruct(finalFields)
      }

    case InsertFields(struct, Seq(), None) => struct
    case InsertFields(SelectFields(old, _), Seq(), Some(insertFieldOrder)) => SelectFields(old, insertFieldOrder)

    case top@Let(x, Let(y, yVal, yBody), xBody) if (x != y) => Let(y, yVal, Let(x, yBody, xBody))

    case l@Let(name, x@InsertFields(old, newFields, fieldOrder), body) if x.typ.size < 500  && {
      val r = Ref(name, x.typ)
      val nfSet = newFields.map(_._1).toSet

      def allRefsCanBePassedThrough(ir1: IR): Boolean = ir1 match {
        case GetField(`r`, fd) => true
        case InsertFields(`r`, inserted, _) => inserted.forall { case (_, toInsert) => allRefsCanBePassedThrough(toInsert) }
        case SelectFields(`r`, fds) => fds.forall(f => !nfSet.contains(f))
        case `r` => false // if the binding is referenced in any other context, don't rewrite
        case _: TableAggregate => true
        case _: MatrixAggregate => true
        case _ => ir1.children
          .zipWithIndex
          .forall {
            case (child: IR, idx) => Binds(ir1, name, idx) || allRefsCanBePassedThrough(child)
            case _ => true
          }
      }

      allRefsCanBePassedThrough(body)
    } =>
      val r = Ref(name, x.typ)
      val fieldNames = newFields.map(_._1).toArray
      val newFieldMap = newFields.toMap
      val newFieldRefs = newFieldMap.map { case (k, ir) =>
        (k, Ref(genUID(), ir.typ))
      } // cannot be mapValues, or genUID() gets run for every usage!
      def copiedNewFieldRefs(): IndexedSeq[(String, IR)] = fieldNames.map(name => (name, newFieldRefs(name).deepCopy())).toFastSeq

      def rewrite(ir1: IR): IR = ir1 match {
        case GetField(Ref(`name`, _), fd) => newFieldRefs.get(fd) match {
          case Some(r) => r.deepCopy()
          case None => GetField(Ref(name, old.typ), fd)
        }
        case ins@InsertFields(Ref(`name`, _), fields, _) =>
          val newFieldSet = fields.map(_._1).toSet
          InsertFields(Ref(name, old.typ),
            copiedNewFieldRefs().filter { case (name, _) => !newFieldSet.contains(name) }
              ++ fields.map { case (name, ir) => (name, rewrite(ir)) },
            Some(ins.typ.fieldNames.toFastSeq))
        case SelectFields(Ref(`name`, _), fds) =>
          SelectFields(InsertFields(Ref(name, old.typ), copiedNewFieldRefs(), Some(x.typ.fieldNames.toFastSeq)), fds)
        case ta: TableAggregate => ta
        case ma: MatrixAggregate => ma
        case _ => ir1.mapChildrenWithIndex {
            case (child: IR, idx) => if (Binds(ir1, name, idx)) child else rewrite(child)
            case (child, _) => child
          }
      }

      val rw = fieldNames.foldLeft[IR](Let(name, old, rewrite(body))) { case (comb, fieldName) =>
        Let(newFieldRefs(fieldName).name, newFieldMap(fieldName), comb)
      }
      ForwardLets[IR](rw)

    case SelectFields(old, fields) if tcoerce[TStruct](old.typ).fieldNames sameElements fields =>
      old

    case SelectFields(SelectFields(old, _), fields) =>
      SelectFields(old, fields)

    case SelectFields(MakeStruct(fields), fieldNames) =>
      val makeStructFields = fields.toMap
      MakeStruct(fieldNames.map(f => f -> makeStructFields(f)))

    case x@SelectFields(InsertFields(struct, insertFields, _), selectFields) =>
      val selectSet = selectFields.toSet
      val insertFields2 = insertFields.filter { case (fName, _) => selectSet.contains(fName) }
      val structSet = struct.typ.asInstanceOf[TStruct].fieldNames.toSet
      val selectFields2 = selectFields.filter(structSet.contains)
      val x2 = InsertFields(SelectFields(struct, selectFields2), insertFields2, Some(selectFields.toFastSeq))
      assert(x2.typ == x.typ)
      x2

    case x@InsertFields(SelectFields(struct, selectFields), insertFields, _) if
    insertFields.exists { case (name, f) => f == GetField(struct, name) } =>
      val fields = x.typ.fieldNames
      val insertNames = insertFields.map(_._1).toSet
      val (oldFields, newFields) =
        insertFields.partition {  case (name, f) => f == GetField(struct, name) }
      val preservedFields = selectFields.filter(f => !insertNames.contains(f)) ++ oldFields.map(_._1)
      InsertFields(SelectFields(struct, preservedFields), newFields, Some(fields.toFastSeq))

    case GetTupleElement(MakeTuple(xs), idx) => xs.find(_._1 == idx).get._2

    case TableCount(MatrixColsTable(child)) if child.columnCount.isDefined => I64(child.columnCount.get)

    case TableCount(child) if child.partitionCounts.isDefined => I64(child.partitionCounts.get.sum)
    case TableCount(CastMatrixToTable(child, _, _)) => TableCount(MatrixRowsTable(child))
    case TableCount(TableMapGlobals(child, _)) => TableCount(child)
    case TableCount(TableMapRows(child, _)) => TableCount(child)
    case TableCount(TableRepartition(child, _, _)) => TableCount(child)
    case TableCount(TableUnion(children)) =>
      children.map(TableCount(_): IR).treeReduce(ApplyBinaryPrimOp(Add(), _, _))
    case TableCount(TableKeyBy(child, _, _)) => TableCount(child)
    case TableCount(TableOrderBy(child, _)) => TableCount(child)
    case TableCount(TableLeftJoinRightDistinct(child, _, _)) => TableCount(child)
    case TableCount(TableIntervalJoin(child, _, _, _)) => TableCount(child)
    case TableCount(TableRange(n, _)) => I64(n)
    case TableCount(TableParallelize(rowsAndGlobal, _)) => Cast(ArrayLen(GetField(rowsAndGlobal, "rows")), TInt64)
    case TableCount(TableRename(child, _, _)) => TableCount(child)
    case TableCount(TableAggregateByKey(child, _)) => TableCount(TableDistinct(child))
    case TableCount(TableExplode(child, path)) =>
      TableAggregate(child,
        ApplyAggOp(
          FastSeq(),
          FastSeq(ArrayLen(CastToArray(path.foldLeft[IR](Ref("row", child.typ.rowType)) { case (comb, s) => GetField(comb, s)})).toL),
          AggSignature(Sum(), FastSeq(), FastSeq(TInt64))))

    case MatrixCount(child) if child.partitionCounts.isDefined || child.columnCount.isDefined =>
      val rowCount = child.partitionCounts match {
        case Some(pc) => I64(pc.sum)
        case None => TableCount(MatrixRowsTable(child))
      }
      val colCount = child.columnCount match {
        case Some(cc) => I32(cc)
        case None => TableCount(MatrixColsTable(child)).toI
      }
      MakeTuple.ordered(FastSeq(rowCount, colCount))
    case MatrixCount(MatrixMapRows(child, _)) => MatrixCount(child)
    case MatrixCount(MatrixMapCols(child,_,  _)) => MatrixCount(child)
    case MatrixCount(MatrixMapEntries(child,_)) => MatrixCount(child)
    case MatrixCount(MatrixFilterEntries(child,_)) => MatrixCount(child)
    case MatrixCount(MatrixAnnotateColsTable(child, _, _)) => MatrixCount(child)
    case MatrixCount(MatrixAnnotateRowsTable(child, _, _, _)) => MatrixCount(child)
    case MatrixCount(MatrixRepartition(child, _, _)) => MatrixCount(child)
    case MatrixCount(MatrixRename(child, _, _, _, _)) => MatrixCount(child)
    case TableCount(TableRead(_, false, r: MatrixBGENReader)) if r.params.includedVariants.isEmpty =>
      I64(r.nVariants)

    // TableGetGlobals should simplify very aggressively
    case TableGetGlobals(child) if child.typ.globalType == TStruct.empty => MakeStruct(FastSeq())
    case TableGetGlobals(TableKeyBy(child, _, _)) => TableGetGlobals(child)
    case TableGetGlobals(TableFilter(child, _)) => TableGetGlobals(child)
    case TableGetGlobals(TableHead(child, _)) => TableGetGlobals(child)
    case TableGetGlobals(TableRepartition(child, _, _)) => TableGetGlobals(child)
    case TableGetGlobals(TableJoin(child1, child2, _, _)) =>
      val g1 = TableGetGlobals(child1)
      val g2 = TableGetGlobals(child2)
      val g1s = genUID()
      val g2s = genUID()
      Let(g1s, g1,
        Let(g2s, g2,
          MakeStruct(
            g1.typ.asInstanceOf[TStruct].fields.map(f => f.name -> (GetField(Ref(g1s, g1.typ), f.name): IR)) ++
              g2.typ.asInstanceOf[TStruct].fields.map(f => f.name -> (GetField(Ref(g2s, g2.typ), f.name): IR)))))
    case TableGetGlobals(x@TableMultiWayZipJoin(children, _, globalName)) =>
      MakeStruct(FastSeq(globalName -> MakeArray(children.map(TableGetGlobals), TArray(children.head.typ.globalType))))
    case TableGetGlobals(TableLeftJoinRightDistinct(child, _, _)) => TableGetGlobals(child)
    case TableGetGlobals(TableMapRows(child, _)) => TableGetGlobals(child)
    case TableGetGlobals(TableMapGlobals(child, newGlobals)) =>
      val uid = genUID()
      val ref = Ref(uid, child.typ.globalType)
      Let(uid, TableGetGlobals(child), Subst(newGlobals, BindingEnv(Env.empty[IR].bind("global", ref))))
    case TableGetGlobals(TableExplode(child, _)) => TableGetGlobals(child)
    case TableGetGlobals(TableUnion(children)) => TableGetGlobals(children.head)
    case TableGetGlobals(TableDistinct(child)) => TableGetGlobals(child)
    case TableGetGlobals(TableAggregateByKey(child, _)) => TableGetGlobals(child)
    case TableGetGlobals(TableKeyByAndAggregate(child, _, _, _, _)) => TableGetGlobals(child)
    case TableGetGlobals(TableOrderBy(child, _)) => TableGetGlobals(child)
    case TableGetGlobals(TableRename(child, _, globalMap)) =>
      if (globalMap.isEmpty)
        TableGetGlobals(child)
      else {
        val uid = genUID()
        val ref = Ref(uid, child.typ.globalType)
        Let(uid, TableGetGlobals(child), MakeStruct(child.typ.globalType.fieldNames.map { f =>
          globalMap.getOrElse(f, f) -> GetField(ref, f)
        }))
      }

    case TableCollect(TableParallelize(x, _)) => x
    case x@TableCollect(TableOrderBy(child, sortFields)) if sortFields.forall(_.sortOrder == Ascending)
      && !child.typ.key.startsWith(sortFields.map(_.field)) =>
      val uid = genUID()
      val uid2 = genUID()
      val left = genUID()
      val right = genUID()
      val uid3 = genUID()
      val sortType = child.typ.rowType.select(sortFields.map(_.field))._1

      val kvElement = MakeStruct(FastSeq(
        ("key", SelectFields(Ref(uid2, child.typ.rowType), sortFields.map(_.field))),
        ("value", Ref(uid2, child.typ.rowType))))
      val sorted = ArraySort(
        StreamMap(
          ToStream(GetField(Ref(uid, x.typ), "rows")),
          uid2,
          kvElement
        ),
        left,
        right,
        ApplyComparisonOp(LT(sortType),
          GetField(Ref(left, kvElement.typ), "key"),
          GetField(Ref(right, kvElement.typ), "key")))
      Let(uid,
        TableCollect(TableKeyBy(child, FastSeq())),
        MakeStruct(FastSeq(
          ("rows", ToArray(StreamMap(ToStream(sorted),
            uid3,
            GetField(Ref(uid3, sorted.typ.asInstanceOf[TArray].elementType), "value")))),
          ("global", GetField(Ref(uid, x.typ), "global")))))
    case ArrayLen(GetField(TableCollect(child), "rows")) => Cast(TableCount(child), TInt32)
    case GetField(TableCollect(child), "global") => TableGetGlobals(child)

    case TableAggregate(child, query) if child.typ.key.nonEmpty && !ContainsNonCommutativeAgg(query) =>
      TableAggregate(TableKeyBy(child, FastSeq(), false), query)
    case TableAggregate(TableOrderBy(child, _), query) if !ContainsNonCommutativeAgg(query) =>
      if (child.typ.key.isEmpty)
        TableAggregate(child, query)
      else
        TableAggregate(TableKeyBy(child, FastSeq(), false), query)
    case TableAggregate(TableMapRows(child, newRow), query) if !ContainsScan(newRow) =>
      val uid = genUID()
      TableAggregate(child,
        AggLet(uid, newRow, Subst(query, BindingEnv(agg = Some(Env("row" -> Ref(uid, newRow.typ))))), isScan = false))

    // NOTE: The below rule should be reintroduced when it is possible to put an ArrayAgg inside a TableAggregate
    // case TableAggregate(TableParallelize(rowsAndGlobal, _), query) =>
    //   rowsAndGlobal match {
    //     // match because we currently don't optimize MakeStruct through Let, and this is a common pattern
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

    case ApplyIR("annotate", _, Seq(s, MakeStruct(fields)), _) =>
      InsertFields(s, fields)

    // simplify Boolean equality
    case ApplyComparisonOp(EQ(_, _), expr, True()) => expr
    case ApplyComparisonOp(EQ(_, _), True(), expr) => expr
    case ApplyComparisonOp(EQ(_, _), expr, False()) => ApplyUnaryPrimOp(Bang(), expr)
    case ApplyComparisonOp(EQ(_, _), False(), expr) => ApplyUnaryPrimOp(Bang(), expr)

    case ApplyUnaryPrimOp(Bang(), ApplyComparisonOp(op, l, r)) =>
      ApplyComparisonOp(ComparisonOp.invert(op.asInstanceOf[ComparisonOp[Boolean]]), l, r)

    case StreamAgg(_, _, query) if {
      def canBeLifted(x: IR): Boolean = x match {
        case _: TableAggregate => true
        case _: MatrixAggregate => true
        case AggLet(_, _, _, false) => false
        case x if IsAggResult(x) => false
        case other => other.children.forall {
          case child: IR => canBeLifted(child)
          case _: BaseIR => true
        }
      }
      canBeLifted(query)
    } => query

    case StreamAggScan(_, _, query) if {
      def canBeLifted(x: IR): Boolean = x match {
        case _: TableAggregate => true
        case _: MatrixAggregate => true
        case AggLet(_, _, _, true) => false
        case x if IsScanResult(x) => false
        case other => other.children.forall {
          case child: IR => canBeLifted(child)
          case _: BaseIR => true
        }
      }
      canBeLifted(query)
    } => query

    case BlockMatrixToValueApply(ValueToBlockMatrix(child, IndexedSeq(nrows, ncols), _), functions.GetElement(Seq(i, j))) => child.typ match {
      case TArray(_) => ArrayRef(child, I32((i * ncols + j).toInt))
      case TNDArray(_, _) => NDArrayRef(child, IndexedSeq(i, j), ErrorIDs.NO_ERROR)
      case TFloat64 => child
    }
    case LiftMeOut(child) if IsConstant(child) => child
  }

  private[this] def tableRules(ctx: ExecuteContext): PartialFunction[TableIR, TableIR] = {

    case TableRename(child, m1, m2) if m1.isTrivial && m2.isTrivial => child

    // TODO: Write more rules like this to bubble 'TableRename' nodes towards the root.
    case t@TableRename(TableKeyBy(child, keys, isSorted), rowMap, globalMap) =>
      TableKeyBy(TableRename(child, rowMap, globalMap), keys.map(t.rowF), isSorted)

    case TableFilter(t, True()) => t

    case TableFilter(TableRead(typ, _, tr), False() | NA(_)) =>
      TableRead(typ, dropRows = true, tr)

    case TableFilter(TableFilter(t, p1), p2) =>
      TableFilter(t,
        ApplySpecial("land", Array.empty[Type], Array(p1, p2), TBoolean, ErrorIDs.NO_ERROR))

    case TableFilter(TableKeyBy(child, key, isSorted), p) =>
      TableKeyBy(TableFilter(child, p), key, isSorted)

    case TableFilter(TableRepartition(child, n, strategy), p) =>
      TableRepartition(TableFilter(child, p), n, strategy)

    case TableOrderBy(TableKeyBy(child, _, false), sortFields) => TableOrderBy(child, sortFields)

    case TableFilter(TableOrderBy(child, sortFields), pred) =>
      TableOrderBy(TableFilter(child, pred), sortFields)

    case TableFilter(TableParallelize(rowsAndGlobal, nPartitions), pred) =>
      val newRowsAndGlobal = rowsAndGlobal match {
        case MakeStruct(Seq(("rows", rows), ("global", globalVal))) =>
          Let("global", globalVal,
            MakeStruct(FastSeq(
              ("rows", ToArray(StreamFilter(ToStream(rows), "row", pred))),
              ("global", Ref("global", globalVal.typ)))))
        case _ =>
          val uid = genUID()
          Let(uid, rowsAndGlobal,
            Let("global", GetField(Ref(uid, rowsAndGlobal.typ), "global"),
              MakeStruct(FastSeq(
                ("rows", ToArray(StreamFilter(ToStream(GetField(Ref(uid, rowsAndGlobal.typ), "rows")), "row", pred))),
                ("global", Ref("global", rowsAndGlobal.typ.asInstanceOf[TStruct].fieldType("global")))
              ))))
      }
      TableParallelize(newRowsAndGlobal, nPartitions)

    case TableKeyBy(TableOrderBy(child, sortFields), keys, false) =>
      TableKeyBy(child, keys, false)

    case TableKeyBy(TableKeyBy(child, _, _), keys, false) =>
      TableKeyBy(child, keys, false)

    case TableKeyBy(TableKeyBy(child, _, true), keys, true) =>
      TableKeyBy(child, keys, true)

    case TableKeyBy(child, key, _) if key == child.typ.key => child

    case TableMapRows(child, Ref("row", _)) => child

    case TableMapRows(child, MakeStruct(fields))
      if fields.length == child.typ.rowType.size
        && fields.zip(child.typ.rowType.fields).forall { case ((_, ir), field) =>
        ir == GetField(Ref("row", field.typ), field.name)
      } =>
      val renamedPairs = for {
        (oldName, (newName, _)) <- child.typ.rowType.fieldNames zip fields
        if oldName != newName
      } yield oldName -> newName
      TableRename(child, Map(renamedPairs: _*), Map.empty)

    case TableMapRows(TableMapRows(child, newRow1), newRow2) if !ContainsScan(newRow2) =>
      TableMapRows(child, Let("row", newRow1, newRow2))

    case TableMapGlobals(child, Ref("global", _)) => child

    // flatten unions
    case TableUnion(children) if children.exists(_.isInstanceOf[TableUnion]) =>
      TableUnion(children.flatMap {
        case u: TableUnion => u.childrenSeq
        case c => Some(c)
      })

    case MatrixRowsTable(MatrixUnionRows(children)) =>
      TableUnion(children.map(MatrixRowsTable))

    case MatrixColsTable(MatrixUnionRows(children)) =>
      MatrixColsTable(children(0))

    // Ignore column or row data that is immediately dropped
    case MatrixRowsTable(MatrixRead(typ, false, dropRows, reader)) =>
      MatrixRowsTable(MatrixRead(typ, dropCols = true, dropRows, reader))

    case MatrixColsTable(MatrixRead(typ, dropCols, false, reader)) =>
      MatrixColsTable(MatrixRead(typ, dropCols, dropRows = true, reader))

    case MatrixRowsTable(MatrixFilterRows(child, pred)) =>
      val mrt = MatrixRowsTable(child)
      TableFilter(
        mrt,
        Subst(pred, BindingEnv(Env("va" -> Ref("row", mrt.typ.rowType)))))

    case MatrixRowsTable(MatrixMapGlobals(child, newGlobals)) => TableMapGlobals(MatrixRowsTable(child), newGlobals)
    case MatrixRowsTable(MatrixMapCols(child, _, _)) => MatrixRowsTable(child)
    case MatrixRowsTable(MatrixMapEntries(child, _)) => MatrixRowsTable(child)
    case MatrixRowsTable(MatrixFilterEntries(child, _)) => MatrixRowsTable(child)
    case MatrixRowsTable(MatrixFilterCols(child, _)) => MatrixRowsTable(child)
    case MatrixRowsTable(MatrixAggregateColsByKey(child, _, _)) => MatrixRowsTable(child)
    case MatrixRowsTable(MatrixChooseCols(child, _)) => MatrixRowsTable(child)
    case MatrixRowsTable(MatrixCollectColsByKey(child)) => MatrixRowsTable(child)
    case MatrixRowsTable(MatrixKeyRowsBy(child, keys, isSorted)) => TableKeyBy(MatrixRowsTable(child), keys, isSorted)

    case MatrixColsTable(x@MatrixMapCols(child, newRow, newKey))
      if newKey.isEmpty
        && !ContainsAgg(newRow)
        && !ContainsScan(newRow) =>
      val mct = MatrixColsTable(child)
      TableMapRows(
        mct,
        Subst(newRow, BindingEnv(Env("sa" -> Ref("row", mct.typ.rowType)))))

    case MatrixColsTable(MatrixMapGlobals(child, newGlobals)) => TableMapGlobals(MatrixColsTable(child), newGlobals)
    case MatrixColsTable(MatrixMapRows(child, _)) => MatrixColsTable(child)
    case MatrixColsTable(MatrixMapEntries(child, _)) => MatrixColsTable(child)
    case MatrixColsTable(MatrixFilterEntries(child, _)) => MatrixColsTable(child)
    case MatrixColsTable(MatrixFilterRows(child, _)) => MatrixColsTable(child)
    case MatrixColsTable(MatrixAggregateRowsByKey(child, _, _)) => MatrixColsTable(child)
    case MatrixColsTable(MatrixKeyRowsBy(child, _, _)) => MatrixColsTable(child)

    case TableRepartition(TableRange(nRows, _), nParts, _) => TableRange(nRows, nParts)

    case TableMapGlobals(TableMapGlobals(child, ng1), ng2) =>
      val uid = genUID()
      TableMapGlobals(child, Let(uid, ng1, Subst(ng2, BindingEnv(Env("global" -> Ref(uid, ng1.typ))))))

    case TableHead(MatrixColsTable(child), n) if child.typ.colKey.isEmpty =>
      if (n > Int.MaxValue) MatrixColsTable(child) else MatrixColsTable(MatrixColsHead(child, n.toInt))

    case TableHead(TableMapRows(child, newRow), n) =>
      TableMapRows(TableHead(child, n), newRow)

    case TableHead(TableRepartition(child, nPar, shuffle), n) =>
      TableRepartition(TableHead(child, n), nPar, shuffle)

    case TableHead(tr@TableRange(nRows, nPar), n) =>
      if (n < nRows)
        TableRange(n.toInt, (nPar.toFloat * n / nRows).toInt.max(1))
      else
        tr

    case TableHead(TableMapGlobals(child, newGlobals), n) =>
      TableMapGlobals(TableHead(child, n), newGlobals)

    case TableHead(TableOrderBy(child, sortFields), n)
      if !TableOrderBy.isAlreadyOrdered(sortFields, child.typ.key) // FIXME: https://github.com/hail-is/hail/issues/6234
        && sortFields.forall(_.sortOrder == Ascending)
        && n < 256 =>
      // n < 256 is arbitrary for memory concerns
      val row = Ref("row", child.typ.rowType)
      val keyStruct = MakeStruct(sortFields.map(f => f.field -> GetField(row, f.field)))
      val aggSig = AggSignature(TakeBy(), FastSeq(TInt32),  FastSeq(row.typ, keyStruct.typ))
      val te =
        TableExplode(
          TableKeyByAndAggregate(child,
            MakeStruct(FastSeq(
              "row" -> ApplyAggOp(
                FastSeq(I32(n.toInt)),
                Array(row, keyStruct),
                aggSig))),
            MakeStruct(FastSeq()), // aggregate to one row
            Some(1), 10),
          FastSeq("row"))
      TableMapRows(te, GetField(Ref("row", te.typ.rowType), "row"))

    case TableDistinct(TableDistinct(child)) => TableDistinct(child)
    case TableDistinct(TableAggregateByKey(child, expr)) => TableAggregateByKey(child, expr)
    case TableDistinct(TableMapRows(child, newRow)) => TableMapRows(TableDistinct(child), newRow)
    case TableDistinct(TableLeftJoinRightDistinct(child, right, root)) => TableLeftJoinRightDistinct(TableDistinct(child), right, root)
    case TableDistinct(TableRepartition(child, n, strategy)) => TableRepartition(TableDistinct(child), n, strategy)

    case TableKeyByAndAggregate(child, MakeStruct(Seq()), k@MakeStruct(keyFields), _, _) =>
      TableDistinct(TableKeyBy(TableMapRows(TableKeyBy(child, FastSeq()), k), k.typ.asInstanceOf[TStruct].fieldNames))

    case TableKeyByAndAggregate(child, expr, newKey, _, _)
      if (newKey == MakeStruct(child.typ.key.map(k => k -> GetField(Ref("row", child.typ.rowType), k))) ||
        newKey == SelectFields(Ref("row", child.typ.rowType), child.typ.key))
        && child.typ.key.nonEmpty =>
      TableAggregateByKey(child, expr)

    case TableAggregateByKey(x@TableKeyBy(child, keys, false), expr) if !x.definitelyDoesNotShuffle =>
      TableKeyByAndAggregate(child, expr, MakeStruct(keys.map(k => k -> GetField(Ref("row", child.typ.rowType), k))), bufferSize = ctx.getFlag("grouped_aggregate_buffer_size").toInt)

    case TableParallelize(TableCollect(child), _) => child

    case TableFilterIntervals(child, intervals, keep) if intervals.isEmpty =>
      if (keep)
        TableFilter(child, False())
      else
        child

    // push down filter intervals nodes
    case TableFilterIntervals(TableFilter(child, pred), intervals, keep) =>
      TableFilter(TableFilterIntervals(child, intervals, keep), pred)
    case TableFilterIntervals(TableMapRows(child, newRow), intervals, keep) if !ContainsScan(newRow) =>
      TableMapRows(TableFilterIntervals(child, intervals, keep), newRow)
    case TableFilterIntervals(TableMapGlobals(child, newRow), intervals, keep) =>
      TableMapGlobals(TableFilterIntervals(child, intervals, keep), newRow)
    case TableFilterIntervals(TableRename(child, rowMap, globalMap), intervals, keep) =>
      TableRename(TableFilterIntervals(child, intervals, keep), rowMap, globalMap)
    case TableFilterIntervals(TableRepartition(child, n, strategy), intervals, keep) =>
      TableRepartition(TableFilterIntervals(child, intervals, keep), n, strategy)
    case TableFilterIntervals(TableLeftJoinRightDistinct(child, right, root), intervals, true) =>
      TableLeftJoinRightDistinct(TableFilterIntervals(child, intervals, true), TableFilterIntervals(right, intervals, true), root)
    case TableFilterIntervals(TableIntervalJoin(child, right, root, product), intervals, keep) =>
      TableIntervalJoin(TableFilterIntervals(child, intervals, keep), right, root, product)
    case TableFilterIntervals(TableJoin(left, right, jt, jk), intervals, true) =>
      TableJoin(TableFilterIntervals(left, intervals, true), TableFilterIntervals(right, intervals, true), jt, jk)
    case TableFilterIntervals(TableExplode(child, path), intervals, keep) =>
      TableExplode(TableFilterIntervals(child, intervals, keep), path)
    case TableFilterIntervals(TableAggregateByKey(child, expr), intervals, keep) =>
      TableAggregateByKey(TableFilterIntervals(child, intervals, keep), expr)
    case TableFilterIntervals(TableFilterIntervals(child, _i1, keep1), _i2, keep2) if keep1 == keep2 =>
      val ord = PartitionBoundOrdering(ctx, child.typ.keyType).intervalEndpointOrdering
      val i1 = Interval.union(_i1.toArray[Interval], ord)
      val i2 = Interval.union(_i2.toArray[Interval], ord)
      val intervals = if (keep1)
      // keep means intersect intervals
        Interval.intersection(i1, i2, ord)
      else
      // remove means union intervals
        Interval.union(i1 ++ i2, ord)
      TableFilterIntervals(child, intervals.toFastSeq, keep1)

      // FIXME: Can try to serialize intervals shorter than the key
      // case TableFilterIntervals(k@TableKeyBy(child, keys, isSorted), intervals, keep) if !child.typ.key.startsWith(keys) =>
      //   val ord = k.typ.keyType.ordering.intervalEndpointOrdering
      //   val maybeFlip: IR => IR = if (keep) identity else !_
      //   val pred = maybeFlip(invoke("sortedNonOverlappingIntervalsContain",
      //     TBoolean,
      //     Literal(TArray(TInterval(k.typ.keyType)), Interval.union(intervals.toArray, ord).toFastIndexedSeq),
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
          NativeReaderOptions(Interval.union(intervals.toArray, PartitionBoundOrdering(ctx, pt).intervalEndpointOrdering), pt, true)
        case Some(NativeReaderOptions(preIntervals, intervalPointType, _)) =>
          val iord = PartitionBoundOrdering(ctx, intervalPointType).intervalEndpointOrdering
          NativeReaderOptions(
            Interval.intersection(Interval.union(preIntervals.toArray, iord), Interval.union(intervals.toArray, iord), iord),
            intervalPointType, true)
      }
      TableRead(t, false, new TableNativeReader(TableNativeReaderParameters(tr.params.path, Some(newOpts)), tr.spec))

    case TableFilterIntervals(TableRead(t, false, tr: TableNativeZippedReader), intervals, true)
      if tr.specLeft.indexed
        && tr.options.forall(_.filterIntervals)
        && SemanticVersion(tr.specLeft.file_version) >= SemanticVersion(1, 3, 0) =>
      val newOpts = tr.options match {
        case None =>
          val pt = t.keyType
          NativeReaderOptions(Interval.union(intervals.toArray, PartitionBoundOrdering(ctx, pt).intervalEndpointOrdering), pt, true)
        case Some(NativeReaderOptions(preIntervals, intervalPointType, _)) =>
          val iord = PartitionBoundOrdering(ctx, intervalPointType).intervalEndpointOrdering
          NativeReaderOptions(
            Interval.intersection(Interval.union(preIntervals.toArray, iord), Interval.union(intervals.toArray, iord), iord),
            intervalPointType, true)
      }
      TableRead(t, false, TableNativeZippedReader(tr.pathLeft, tr.pathRight, Some(newOpts), tr.specLeft, tr.specRight))
  }

  private[this] def matrixRules(): PartialFunction[MatrixIR, MatrixIR] = {
    case MatrixMapRows(child, Ref("va", _)) => child

    case MatrixKeyRowsBy(MatrixKeyRowsBy(child, _, _), keys, false) =>
      MatrixKeyRowsBy(child, keys, false)

    case MatrixKeyRowsBy(MatrixKeyRowsBy(child, _, true), keys, true) =>
      MatrixKeyRowsBy(child, keys, true)

    case MatrixMapCols(child, Ref("sa", _), None) => child

    case x@MatrixMapEntries(child, Ref("g", _)) =>
      assert(child.typ == x.typ)
      child

    case x@MatrixMapEntries(MatrixMapEntries(child, newEntries1), newEntries2) =>
      val uid = genUID()
      val ne2 = Subst(newEntries2, BindingEnv(Env("g" -> Ref(uid, newEntries1.typ))))
      MatrixMapEntries(child, Let(uid, newEntries1, ne2))

    case MatrixMapGlobals(child, Ref("global", _)) => child

    // flatten unions
    case MatrixUnionRows(children) if children.exists(_.isInstanceOf[MatrixUnionRows]) =>
      MatrixUnionRows(children.flatMap {
        case u: MatrixUnionRows => u.childrenSeq
        case c => Some(c)
      })

    // Equivalent rewrites for the new Filter{Cols,Rows}IR
    case MatrixFilterRows(MatrixRead(typ, dropCols, _, reader), False() | NA(_)) =>
      MatrixRead(typ, dropCols, dropRows = true, reader)

    case MatrixFilterCols(MatrixRead(typ, _, dropRows, reader), False() | NA(_)) =>
      MatrixRead(typ, dropCols = true, dropRows, reader)

    // Keep all rows/cols = do nothing
    case MatrixFilterRows(m, True()) => m

    case MatrixFilterCols(m, True()) => m

    case MatrixFilterRows(MatrixFilterRows(child, pred1), pred2) => MatrixFilterRows(child, ApplySpecial("land", FastSeq(), FastSeq(pred1, pred2), TBoolean, ErrorIDs.NO_ERROR))

    case MatrixFilterCols(MatrixFilterCols(child, pred1), pred2) => MatrixFilterCols(child, ApplySpecial("land", FastSeq(), FastSeq(pred1, pred2), TBoolean, ErrorIDs.NO_ERROR))

    case MatrixFilterEntries(MatrixFilterEntries(child, pred1), pred2) => MatrixFilterEntries(child, ApplySpecial("land", FastSeq(), FastSeq(pred1, pred2), TBoolean, ErrorIDs.NO_ERROR))

    case MatrixMapGlobals(MatrixMapGlobals(child, ng1), ng2) =>
      val uid = genUID()
      MatrixMapGlobals(child, Let(uid, ng1, Subst(ng2, BindingEnv(Env("global" -> Ref(uid, ng1.typ))))))

    // Note: the following MMR and MMC fusing rules are much weaker than they could be. If they contain aggregations
    // but those aggregations that mention "row" / "sa" but do not depend on the updated value, we should locally
    // prune and fuse anyway.
    case MatrixMapRows(MatrixMapRows(child, newRow1), newRow2) if !Mentions.inAggOrScan(newRow2, "va")
      && !Exists.inIR(newRow2, {
      case a: ApplyAggOp => a.initOpArgs.exists(Mentions(_, "va")) // Lowering produces invalid IR
      case _ => false
    }) =>
      val uid = genUID()
      MatrixMapRows(child, Let(uid, newRow1,
        Subst(newRow2, BindingEnv[IR](Env(("va", Ref(uid, newRow1.typ))),
          agg = Some(Env.empty[IR]),
          scan = Some(Env.empty[IR])))))

    case MatrixMapCols(MatrixMapCols(child, newCol1, nk1), newCol2, nk2) if !Mentions.inAggOrScan(newCol2, "sa") =>
      val uid = genUID()
      MatrixMapCols(child, Let(uid, newCol1,
        Subst(newCol2, BindingEnv[IR](Env(("sa", Ref(uid, newCol1.typ))),
          agg = Some(Env.empty[IR]),
          scan = Some(Env.empty[IR])))),
        if (nk2.isDefined) nk2 else nk1)

    // bubble up MatrixColsHead node
    case MatrixColsHead(MatrixMapCols(child, newCol, newKey), n) => MatrixMapCols(MatrixColsHead(child, n), newCol, newKey)
    case MatrixColsHead(MatrixMapEntries(child, newEntries), n) => MatrixMapEntries(MatrixColsHead(child, n), newEntries)
    case MatrixColsHead(MatrixFilterEntries(child, newEntries), n) => MatrixFilterEntries(MatrixColsHead(child, n), newEntries)
    case MatrixColsHead(MatrixKeyRowsBy(child, keys, isSorted), n) => MatrixKeyRowsBy(MatrixColsHead(child, n), keys, isSorted)
    case MatrixColsHead(MatrixAggregateRowsByKey(child, rowExpr, entryExpr), n) => MatrixAggregateRowsByKey(MatrixColsHead(child, n), rowExpr, entryExpr)
    case MatrixColsHead(MatrixChooseCols(child, oldIndices), n) => MatrixChooseCols(child, oldIndices.take(n))
    case MatrixColsHead(MatrixColsHead(child, n1), n2) => MatrixColsHead(child, math.min(n1, n2))
    case MatrixColsHead(MatrixFilterRows(child, pred), n) => MatrixFilterRows(MatrixColsHead(child, n), pred)
    case MatrixColsHead(MatrixRead(t, dr, dc, r: MatrixRangeReader), n) =>
      MatrixRead(t, dr, dc, MatrixRangeReader(r.params.nRows, math.min(r.params.nCols, n), r.params.nPartitions))
    case MatrixColsHead(MatrixMapRows(child, newRow), n) if !Mentions.inAggOrScan(newRow, "sa") =>
      MatrixMapRows(MatrixColsHead(child, n), newRow)
    case MatrixColsHead(MatrixMapGlobals(child, newGlobals), n) => MatrixMapGlobals(MatrixColsHead(child, n), newGlobals)
    case MatrixColsHead(MatrixAnnotateColsTable(child, table, root), n) => MatrixAnnotateColsTable(MatrixColsHead(child, n), table, root)
    case MatrixColsHead(MatrixAnnotateRowsTable(child, table, root, product), n) => MatrixAnnotateRowsTable(MatrixColsHead(child, n), table, root, product)
    case MatrixColsHead(MatrixRepartition(child, nPar, strategy), n) => MatrixRepartition(MatrixColsHead(child, n), nPar, strategy)
    case MatrixColsHead(MatrixExplodeRows(child, path), n) => MatrixExplodeRows(MatrixColsHead(child, n), path)
    case MatrixColsHead(MatrixUnionRows(children), n) =>
      // could prevent a dimension mismatch error, but we view errors as undefined behavior, so this seems OK.
      MatrixUnionRows(children.map(MatrixColsHead(_, n)))
    case MatrixColsHead(MatrixDistinctByRow(child), n) => MatrixDistinctByRow(MatrixColsHead(child, n))
    case MatrixColsHead(MatrixRename(child, glob, col, row, entry), n) => MatrixRename(MatrixColsHead(child, n), glob, col, row, entry)
  }

  private[this] def blockMatrixRules: PartialFunction[BlockMatrixIR, BlockMatrixIR] = {
    case BlockMatrixBroadcast(child, IndexedSeq(0, 1), _, _) => child
    case BlockMatrixSlice(BlockMatrixMap(child, n, f, reqDense), slices) => BlockMatrixMap(BlockMatrixSlice(child, slices), n, f, reqDense)
    case BlockMatrixSlice(BlockMatrixMap2(l, r, ln, rn, f, sparsityStrategy), slices) =>
      BlockMatrixMap2(BlockMatrixSlice(l, slices), BlockMatrixSlice(r, slices), ln, rn, f, sparsityStrategy)
    case BlockMatrixMap2(BlockMatrixBroadcast(scalarBM, IndexedSeq(), _, _), right, leftName, rightName, f, sparsityStrategy) =>
      val getElement = BlockMatrixToValueApply(scalarBM, functions.GetElement(IndexedSeq(0, 0)))
      val needsDense = sparsityStrategy == NeedsDense || sparsityStrategy.exists(leftBlock = true, rightBlock = false)
      val maybeDense = if (needsDense) BlockMatrixDensify(right) else right
      BlockMatrixMap(maybeDense, rightName, Subst(f, BindingEnv.eval(leftName -> getElement)), needsDense)
    case BlockMatrixMap2(left, BlockMatrixBroadcast(scalarBM, IndexedSeq(), _, _), leftName, rightName, f, sparsityStrategy) =>
      val getElement = BlockMatrixToValueApply(scalarBM, functions.GetElement(IndexedSeq(0, 0)))
      val needsDense = sparsityStrategy == NeedsDense || sparsityStrategy.exists(leftBlock = false, rightBlock = true)
      val maybeDense = if (needsDense) BlockMatrixDensify(left) else left
      BlockMatrixMap(maybeDense, leftName, Subst(f, BindingEnv.eval(rightName -> getElement)), needsDense)
    case BlockMatrixMap(matrix, name, Ref(x, _), _) if name == x =>
      matrix
    case BlockMatrixMap(matrix, name, ir, _) if IsConstant(ir) || (ir.isInstanceOf[Ref] && ir.asInstanceOf[Ref].name != name) =>
      val typ = matrix.typ
      BlockMatrixBroadcast(
        ValueToBlockMatrix(ir, FastSeq(1, 1), typ.blockSize),
        FastSeq(),
        typ.shape,
        typ.blockSize
      )
  }
}

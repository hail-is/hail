package is.hail.expr.ir

import is.hail.expr.types.virtual._
import is.hail.io.bgen.MatrixBGENReader
import is.hail.table.Ascending
import is.hail.utils._

object Simplify {

  /** Transform 'ir' using simplification rules until none apply.
    */
  def apply(ir: BaseIR): BaseIR = Simplify(ir, allowRepartitioning = true)

  /** Use 'allowRepartitioning'=false when in a context where simplification
    * should not change the partitioning of the result of 'ast', such as when
    * some parent (downstream) node of 'ast' uses seeded randomness.
    */
  private[ir] def apply(ast: BaseIR, allowRepartitioning: Boolean): BaseIR =
    ast match {
      case ir: IR => simplifyValue(ir)
      case tir: TableIR => simplifyTable(allowRepartitioning)(tir)
      case mir: MatrixIR => simplifyMatrix(allowRepartitioning)(mir)
      case bmir: BlockMatrixIR => simplifyBlockMatrix(bmir)
    }

  private[this] def visitNode[T <: BaseIR](
    visitChildren: BaseIR => BaseIR,
    transform: T => Option[T],
    post: => (T => T)
  )(t: T): T = {
    val t1 = t.mapChildren(visitChildren).asInstanceOf[T]
    transform(t1).map(post).getOrElse(t1)
  }

  private[this] def simplifyValue: IR => IR =
    visitNode(
      Simplify(_),
      rewriteValueNode,
      simplifyValue)

  private[this] def simplifyTable(allowRepartitioning: Boolean)(tir: TableIR): TableIR =
    visitNode(
      Simplify(_, allowRepartitioning && isDeterministicallyRepartitionable(tir)),
      rewriteTableNode(allowRepartitioning),
      simplifyTable(allowRepartitioning)
    )(tir)

  private[this] def simplifyMatrix(allowRepartitioning: Boolean)(mir: MatrixIR): MatrixIR =
    visitNode(
      Simplify(_, allowRepartitioning && isDeterministicallyRepartitionable(mir)),
      rewriteMatrixNode(allowRepartitioning),
      simplifyMatrix(allowRepartitioning)
    )(mir)

  private[this] def simplifyBlockMatrix(bmir: BlockMatrixIR): BlockMatrixIR = {
    visitNode(
      Simplify(_),
      rewriteBlockMatrixNode,
      simplifyBlockMatrix
    )(bmir)
  }

  private[this] def rewriteValueNode: IR => Option[IR] = valueRules.lift

  private[this] def rewriteTableNode(allowRepartitioning: Boolean)(tir: TableIR): Option[TableIR] =
    tableRules(allowRepartitioning && isDeterministicallyRepartitionable(tir)).lift(tir)

  private[this] def rewriteMatrixNode(allowRepartitioning: Boolean)(mir: MatrixIR): Option[MatrixIR] =
    matrixRules(allowRepartitioning && isDeterministicallyRepartitionable(mir)).lift(mir)

  private[this] def rewriteBlockMatrixNode: BlockMatrixIR => Option[BlockMatrixIR] = blockMatrixRules.lift

  /** Returns true if 'x' propagates missingness, meaning if any child of 'x'
    * evaluates to missing, then 'x' will evaluate to missing.
    */
  private[this] def isStrict(x: IR): Boolean = {
    x match {
      case _: Apply |
           _: ApplyUnaryPrimOp |
           _: ApplyBinaryPrimOp |
           _: ArrayRange |
           _: ArrayRef |
           _: ArrayLen |
           _: GetField |
           _: GetTupleElement => true
      case ApplyComparisonOp(op, _, _) => op.strict
      case f: ApplySeeded => f.implementation.isStrict
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

  /** Returns true if changing the partitioning of child (upstream) nodes of
    * 'ir' can not otherwise change the contents of the result of 'ir'.
    */
  private[this] def isDeterministicallyRepartitionable(ir: BaseIR): Boolean =
    ir.children.forall {
      case child: IR => !Exists(child, _.isInstanceOf[ApplySeeded])
      case _ => true
    }

  private[this] def areFieldSelects(fields: Seq[(String, IR)]): Boolean = {
    assert(fields.nonEmpty)
    fields.head match {
      case (_, GetField(s1, _)) =>
        fields.forall {
          case (f1, GetField(s2, f2)) if f1 == f2 && s1 == s2 => true
          case _ => false
        }
      case _ => false
    }
  }

  private[this] def valueRules: PartialFunction[IR, IR] = {
    // propagate NA
    case x: IR if isStrict(x) && Children(x).exists(_.isInstanceOf[NA]) =>
      NA(x.typ)

    case x@If(NA(_), _, _) => NA(x.typ)

    case Coalesce(values) if isDefinitelyDefined(values.head) => values.head

    case Coalesce(values) if values.zipWithIndex.exists { case (ir, i) => isDefinitelyDefined(ir) && i != values.size - 1 } =>
      val idx = values.indexWhere(isDefinitelyDefined)
      Coalesce(values.take(idx + 1))

    case Coalesce(values) if values.size == 1 => values.head

    case x@ArrayMap(NA(_), _, _) => NA(x.typ)

    case x@ArrayFlatMap(NA(_), _, _) => NA(x.typ)

    case x@ArrayFilter(NA(_), _, _) => NA(x.typ)

    case x@ArrayFold(NA(_), _, _, _, _) => NA(x.typ)

    case IsNA(NA(_)) => True()

    case IsNA(x) if isDefinitelyDefined(x) => False()

    case x@If(True(), cnsq, _) if x.typ == cnsq.typ => cnsq

    case x@If(False(), _, altr) if x.typ == altr.typ => altr

    case If(c, cnsq, altr) if cnsq == altr =>
      if (cnsq.typ.required)
        cnsq
      else
        If(IsNA(c), NA(cnsq.typ), cnsq)

    case If(ApplyUnaryPrimOp(Bang(), c), cnsq, altr) => If(c, altr, cnsq)

    case If(c1, If(c2, cnsq2, _), altr1) if c1 == c2 => If(c1, cnsq2, altr1)

    case If(c1, cnsq1, If(c2, _, altr2)) if c1 == c2 => If(c1, cnsq1, altr2)

    case Cast(x, t) if x.typ == t => x

    case CastRename(x, t) if x.typ == t => x

    case ApplyBinaryPrimOp(Add(), I32(0), x) => x
    case ApplyBinaryPrimOp(Add(), x, I32(0)) => x
    case ApplyBinaryPrimOp(Subtract(), I32(0), x) => x
    case ApplyBinaryPrimOp(Subtract(), x, I32(0)) => x

    case ApplyIR("indexArray", Seq(a, i@I32(v))) if v >= 0 =>
      ArrayRef(a, i)

    case ToArray(x) if x.typ.isInstanceOf[TArray] => x

    case ApplyIR("contains", Seq(ToArray(x), element)) if x.typ.isInstanceOf[TSet] => invoke("contains", TBoolean(), x, element)

    case ApplyIR("contains", Seq(Literal(t, v), element)) if t.isInstanceOf[TArray] =>
      invoke("contains", TBoolean(), Literal(TSet(t.asInstanceOf[TArray].elementType, t.required), v.asInstanceOf[IndexedSeq[_]].toSet), element)

    case ApplyIR("contains", Seq(ToSet(x), element)) if x.typ.isInstanceOf[TArray] => invoke("contains", TBoolean(), x, element)

    case x: ApplyIR if x.body.size < 10 => x.explicitNode

    case ArrayLen(MakeArray(args, _)) => I32(args.length)

    case ArrayLen(ArrayRange(start, end, I32(1))) => ApplyBinaryPrimOp(Subtract(), end, start)

    case ArrayLen(ArrayMap(a, _, _)) => ArrayLen(a)

    case ArrayLen(ArrayFlatMap(a, _, MakeArray(args, _))) => ApplyBinaryPrimOp(Multiply(), I32(args.length), ArrayLen(a))

    case ArrayLen(ArraySort(a, _, _, _)) => ArrayLen(a)

    case ArrayRef(MakeArray(args, _), I32(i)) if i >= 0 && i < args.length => args(i)

    case ArrayFilter(a, _, True()) => a

    case ArrayFor(_, _, Begin(Seq())) => Begin(FastIndexedSeq())

    case ArrayFold(ArrayMap(a, n1, b), zero, accumName, valueName, body) => ArrayFold(a, zero, accumName, n1, Let(valueName, b, body))

    case ArrayFlatMap(ArrayMap(a, n1, b1), n2, b2) =>
      ArrayFlatMap(a, n1, Let(n2, b1, b2))

    case ArrayMap(a, elt, r: Ref) if r.name == elt && r.typ == a.typ.asInstanceOf[TArray].elementType => a

    case ArrayMap(ArrayMap(a, n1, b1), n2, b2) =>
      ArrayMap(a, n1, Let(n2, b1, b2))

    case NDArrayShape(MakeNDArray(_, shape, _)) => shape

    case NDArrayShape(NDArrayMap(nd, _, _)) => NDArrayShape(nd)

    case GetField(MakeStruct(fields), name) =>
      val (_, x) = fields.find { case (n, _) => n == name }.get
      x

    case GetField(InsertFields(old, fields, _), name) =>
      fields.find { case (n, _) => n == name } match {
        case Some((_, x)) => x
        case None => GetField(old, name)
      }

    case GetField(SelectFields(old, fields), name) => GetField(old, name)

    case InsertFields(InsertFields(base, fields1, fieldOrder1), fields2, fieldOrder2) =>
        val fields2Set = fields2.map(_._1).toSet
        val newFields = fields1.filter { case (name, _) => !fields2Set.contains(name) } ++ fields2
      (fieldOrder1, fieldOrder2) match {
        case (Some(fo1), None) =>
          val fields1Set = fo1.toSet
          val fieldOrder = fo1 ++ fields2.map(_._1).filter(!fields1Set.contains(_))
          InsertFields(base, newFields, Some(fieldOrder))
        case _ =>
          InsertFields(base, newFields, fieldOrder2)
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


    case InsertFields(struct, Seq(), _) => struct

    case Let(x, InsertFields(parentRef: Ref, insFields, ord1), InsertFields(Ref(x2, _), fields, ord2)) if x2 == x && {
      val insFieldSet = insFields.map(_._1).toSet

      def allRefsCanBePassedThrough(ir1: IR, newBindingEnv: BindingEnv[Type]): Boolean = ir1 match {
        case GetField(Ref(`x`, _), fd) if !insFieldSet.contains(fd) => true
        case Ref(`x`, _) => newBindingEnv.eval.lookupOption(x).isDefined
        case _: TableAggregate => true
        case _: MatrixAggregate => true
        case _ => ir1.children
          .iterator
          .zipWithIndex
          .forall {
            case (child: IR, idx) =>
              allRefsCanBePassedThrough(child, ChildBindings(ir1, idx, newBindingEnv))
            case _ => true
          }
      }

      val baseEnv = BindingEnv(Env.empty[Type], Some(Env.empty[Type]), Some(Env.empty[Type]))
      fields.forall { case (_, ir) =>
        allRefsCanBePassedThrough(ir, baseEnv)
      }
    } =>
      val e = Env[IR]((x, parentRef))
      Subst(
        InsertFields(InsertFields(parentRef, insFields, ord1), fields, ord2),
        BindingEnv(e, Some(e), Some(e)))

    case SelectFields(old, fields) if coerce[TStruct](old.typ).fieldNames sameElements fields =>
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
      val x2 = InsertFields(SelectFields(struct, selectFields2), insertFields2, Some(selectFields.toFastIndexedSeq))
      assert(x2.typ == x.typ)
      x2

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
    case TableCount(TableParallelize(rowsAndGlobal, _)) => Cast(ArrayLen(GetField(rowsAndGlobal, "rows")), TInt64())
    case TableCount(TableRename(child, _, _)) => TableCount(child)
    case TableCount(TableAggregateByKey(child, _)) => TableCount(TableDistinct(child))
    case TableCount(TableExplode(child, path)) =>
      TableAggregate(child,
        ApplyAggOp(
          FastIndexedSeq(),
          None,
          FastIndexedSeq(ArrayLen(ToArray(path.foldLeft[IR](Ref("row", child.typ.rowType)) { case (comb, s) => GetField(comb, s)})).toL),
          AggSignature(Sum(), FastSeq(), None, FastSeq(TInt64()))))

    case TableCount(TableRead(_, false, r: MatrixBGENReader)) if r.includedVariants.isEmpty =>
      I64(r.fileMetadata.map(_.nVariants).sum)

    // TableGetGlobals should simplify very aggressively
    case TableGetGlobals(child) if child.typ.globalType == TStruct() => MakeStruct(FastSeq())
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
    case TableGetGlobals(TableZipUnchecked(left, _)) => TableGetGlobals(left)
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
    case ArrayLen(GetField(TableCollect(child), "rows")) => TableCount(child)

    case TableAggregate(TableMapRows(child, newRow), query) if !ContainsScan(newRow) =>
      val uid = genUID()
      TableAggregate(child,
        AggLet(uid, newRow, Subst(query, BindingEnv(agg = Some(Env("row" -> Ref(uid, newRow.typ))))), isScan = false))

    case ApplyIR("annotate", Seq(s, MakeStruct(fields))) =>
      InsertFields(s, fields)

    // simplify Boolean equality
    case ApplyComparisonOp(EQ(_, _), expr, True()) => expr
    case ApplyComparisonOp(EQ(_, _), True(), expr) => expr
    case ApplyComparisonOp(EQ(_, _), expr, False()) => ApplyUnaryPrimOp(Bang(), expr)
    case ApplyComparisonOp(EQ(_, _), False(), expr) => ApplyUnaryPrimOp(Bang(), expr)

    case ApplyUnaryPrimOp(Bang(), ApplyComparisonOp(op, l, r)) =>
      ApplyComparisonOp(ComparisonOp.invert(op.asInstanceOf[ComparisonOp[Boolean]]), l, r)
  }

  private[this] def tableRules(canRepartition: Boolean): PartialFunction[TableIR, TableIR] = {

    case TableRename(child, m1, m2) if m1.isTrivial && m2.isTrivial => child

    // TODO: Write more rules like this to bubble 'TableRename' nodes towards the root.
    case t@TableRename(TableKeyBy(child, keys, isSorted), rowMap, globalMap) =>
      TableKeyBy(TableRename(child, rowMap, globalMap), keys.map(t.rowF), isSorted)

    case TableFilter(t, True()) => t

    case TableFilter(TableRead(typ, _, tr), False() | NA(_)) =>
      TableRead(typ, dropRows = true, tr)

    case TableFilter(TableFilter(t, p1), p2) =>
      TableFilter(t,
        ApplySpecial("&&", Array(p1, p2), TBoolean()))

    case TableFilter(TableKeyBy(child, key, isSorted), p) if canRepartition => TableKeyBy(TableFilter(child, p), key, isSorted)
    case TableFilter(TableRepartition(child, n, strategy), p) => TableRepartition(TableFilter(child, p), n, strategy)

    case TableOrderBy(TableKeyBy(child, _, _), sortFields) => TableOrderBy(child, sortFields)

    case TableFilter(TableOrderBy(child, sortFields), pred) if canRepartition =>
      TableOrderBy(TableFilter(child, pred), sortFields)

    case TableKeyBy(TableOrderBy(child, sortFields), keys, false) if canRepartition =>
      TableKeyBy(child, keys, false)

    case TableKeyBy(TableKeyBy(child, _, _), keys, false) if canRepartition =>
      TableKeyBy(child, keys, false)

    case TableKeyBy(TableKeyBy(child, _, true), keys, true) if canRepartition =>
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
    case TableUnion(children) if children.exists(_.isInstanceOf[TableUnion]) & canRepartition =>
      TableUnion(children.flatMap {
        case u: TableUnion => u.children
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
        && canRepartition
        && isDeterministicallyRepartitionable(x)
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

    case TableMapGlobals(TableMapGlobals(child, ng1), ng2) =>
      val uid = genUID()
      TableMapGlobals(child, Let(uid, ng1, Subst(ng2, BindingEnv(Env("global" -> Ref(uid, ng1.typ))))))

    case TableHead(MatrixColsTable(child), n) if child.typ.colKey.isEmpty =>
      if (n > Int.MaxValue) MatrixColsTable(child) else MatrixColsTable(MatrixColsHead(child, n.toInt))

    case TableHead(TableMapRows(child, newRow), n) =>
      TableMapRows(TableHead(child, n), newRow)

    case TableHead(TableRepartition(child, nPar, shuffle), n) if canRepartition =>
      TableRepartition(TableHead(child, n), nPar, shuffle)

    case TableHead(tr@TableRange(nRows, nPar), n) if canRepartition =>
      if (n < nRows)
        TableRange(n.toInt, (nPar.toFloat * n / nRows).toInt.max(1))
      else
        tr

    case TableHead(TableMapGlobals(child, newGlobals), n) =>
      TableMapGlobals(TableHead(child, n), newGlobals)

    case TableHead(TableOrderBy(child, sortFields), n)
      if !TableOrderBy.isAlreadyOrdered(sortFields, child.typ.key) // FIXME: https://github.com/hail-is/hail/issues/6234
        && sortFields.forall(_.sortOrder == Ascending)
        && n < 256 && canRepartition =>
      // n < 256 is arbitrary for memory concerns
      val row = Ref("row", child.typ.rowType)
      val keyStruct = MakeStruct(sortFields.map(f => f.field -> GetField(row, f.field)))
      val aggSig = AggSignature(TakeBy(), FastSeq(TInt32()), None, FastSeq(row.typ, keyStruct.typ))
      val te =
        TableExplode(
          TableKeyByAndAggregate(child,
            MakeStruct(Seq(
              "row" -> ApplyAggOp(
                FastIndexedSeq(I32(n.toInt)),
                None,
                Array(row, keyStruct),
                aggSig))),
            MakeStruct(Seq()), // aggregate to one row
            Some(1), 10),
          FastIndexedSeq("row"))
      TableMapRows(te, GetField(Ref("row", te.typ.rowType), "row"))

    case TableDistinct(TableDistinct(child)) => TableDistinct(child)
    case TableDistinct(TableAggregateByKey(child, expr)) => TableAggregateByKey(child, expr)
    case TableDistinct(TableMapRows(child, newRow)) => TableMapRows(TableDistinct(child), newRow)
    case TableDistinct(TableLeftJoinRightDistinct(child, right, root)) => TableLeftJoinRightDistinct(TableDistinct(child), right, root)
    case TableDistinct(TableRepartition(child, n, strategy)) if canRepartition => TableRepartition(TableDistinct(child), n, strategy)

    case TableKeyByAndAggregate(child, MakeStruct(Seq()), k@MakeStruct(keyFields), _, _) if canRepartition =>
      TableDistinct(TableKeyBy(TableMapRows(TableKeyBy(child, FastIndexedSeq()), k), k.typ.asInstanceOf[TStruct].fieldNames))

    case TableKeyByAndAggregate(child, expr, newKey, _, _)
      if newKey == MakeStruct(child.typ.key.map(k => k -> GetField(Ref("row", child.typ.rowType), k)))
        && child.typ.key.nonEmpty && canRepartition =>
      TableAggregateByKey(child, expr)

    case TableAggregateByKey(TableKeyBy(child, keys, _), expr) if canRepartition =>
      TableKeyByAndAggregate(child, expr, MakeStruct(keys.map(k => k -> GetField(Ref("row", child.typ.rowType), k))))

    case TableParallelize(TableCollect(child), _) if isDeterministicallyRepartitionable(child) => child

    case TableZipUnchecked(left, right) if left.typ.rowType.size == 0 =>
      if (left.typ.globalType.size == 0)
        right
      else
        TableMapGlobals(right, TableGetGlobals(left))

    // push down filter intervals nodes
    case TableFilterIntervals(TableFilter(child, pred), intervals, keep) =>
      TableFilter(TableFilterIntervals(child, intervals, keep), pred)
    case TableFilterIntervals(TableMapRows(child, newRow), intervals, keep) if !ContainsScan(newRow) =>
      TableMapRows(TableFilterIntervals(child, intervals, keep), newRow)
    case TableFilterIntervals(TableMapGlobals(child, newRow), intervals, keep) =>
      TableMapGlobals(TableFilterIntervals(child, intervals, keep), newRow)
    case TableFilterIntervals(TableRepartition(child, n, strategy), intervals, keep) =>
      TableRepartition(TableFilterIntervals(child, intervals, keep), n, strategy)
    case TableFilterIntervals(TableLeftJoinRightDistinct(child, right, root), intervals, true) =>
      TableLeftJoinRightDistinct(TableFilterIntervals(child, intervals, true), TableFilterIntervals(right, intervals, true), root)
    case TableFilterIntervals(TableIntervalJoin(child, right, root, product), intervals, keep) =>
      TableIntervalJoin(TableFilterIntervals(child, intervals, keep), right, root, product)
    case TableFilterIntervals(TableExplode(child, path), intervals, keep) =>
      TableExplode(TableFilterIntervals(child, intervals, keep), path)
    case TableFilterIntervals(TableAggregateByKey(child, expr), intervals, keep) =>
      TableAggregateByKey(TableFilterIntervals(child, intervals, keep), expr)
    case TableFilterIntervals(TableFilterIntervals(child, i1, keep1), i2, keep2) if keep1 == keep2 =>
      val ord = child.typ.keyType.ordering.intervalEndpointOrdering
      val intervals = if (keep1)
      // keep means intersect intervals
        Interval.intersection(i1.toArray[Interval], i2.toArray[Interval], ord)
      else
      // remove means union intervals
        Interval.union(i1.toArray[Interval] ++ i2.toArray[Interval], ord)
      TableFilterIntervals(child, intervals.toFastIndexedSeq, keep1)
  }

  private[this] def matrixRules(canRepartition: Boolean): PartialFunction[MatrixIR, MatrixIR] = {
    case MatrixMapRows(child, Ref("va", _)) => child

    case MatrixKeyRowsBy(MatrixKeyRowsBy(child, _, _), keys, false) if canRepartition =>
      MatrixKeyRowsBy(child, keys, false)

    case MatrixKeyRowsBy(MatrixKeyRowsBy(child, _, true), keys, true) if canRepartition =>
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
    case MatrixUnionRows(children) if children.exists(_.isInstanceOf[MatrixUnionRows]) & canRepartition =>
      MatrixUnionRows(children.flatMap {
        case u: MatrixUnionRows => u.children
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

    case MatrixFilterRows(MatrixFilterRows(child, pred1), pred2) => MatrixFilterRows(child, ApplySpecial("&&", FastSeq(pred1, pred2), TBoolean()))

    case MatrixFilterCols(MatrixFilterCols(child, pred1), pred2) => MatrixFilterCols(child, ApplySpecial("&&", FastSeq(pred1, pred2), TBoolean()))

    case MatrixFilterEntries(MatrixFilterEntries(child, pred1), pred2) => MatrixFilterEntries(child, ApplySpecial("&&", FastSeq(pred1, pred2), TBoolean()))

    case MatrixMapGlobals(MatrixMapGlobals(child, ng1), ng2) =>
      val uid = genUID()
      MatrixMapGlobals(child, Let(uid, ng1, Subst(ng2, BindingEnv(Env("global" -> Ref(uid, ng1.typ))))))

    // Note: the following MMR and MMC fusing rules are much weaker than they could be. If they contain aggregations
    // but those aggregations that mention "row" / "sa" but do not depend on the updated value, we should locally
    // prune and fuse anyway.
    case MatrixMapRows(MatrixMapRows(child, newRow1), newRow2) if !Mentions.inAggOrScan(newRow2, "va")
      && !Exists.inIR(newRow2, {
      case a: ApplyAggOp => a.initOpArgs.exists(_.exists(Mentions(_, "va"))) // Lowering produces invalid IR
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
    case MatrixColsHead(MatrixRead(t, dr, dc, MatrixRangeReader(nRows, nCols, nPartitions)), n) =>
      MatrixRead(t, dr, dc, MatrixRangeReader(nRows, math.min(nCols, n), nPartitions))
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
    case BlockMatrixSlice(BlockMatrixMap(child, f), slices) => BlockMatrixMap(BlockMatrixSlice(child, slices), f)
    case BlockMatrixSlice(BlockMatrixMap2(l, r, f), slices) =>
      BlockMatrixMap2(BlockMatrixSlice(l, slices), BlockMatrixSlice(r, slices), f)
  }
}

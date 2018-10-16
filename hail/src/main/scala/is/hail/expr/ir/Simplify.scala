package is.hail.expr.ir

import is.hail.utils._
import is.hail.expr._
import is.hail.expr.types.{TArray, TInt32, TInt64, TStruct}
import is.hail.table.{Ascending, SortField}

object Simplify {

  def apply(ir: BaseIR): BaseIR = Simplify(ir, allowRepartitioning = true)

  private[ir] def apply(ast: BaseIR, allowRepartitioning: Boolean): BaseIR = {
    ast match {
      case ir: IR => simplifyValue(ir)
      case tir: TableIR => simplifyTable(allowRepartitioning)(tir)
      case mir: MatrixIR => simplifyMatrix(allowRepartitioning)(mir)
    }
  }

  private[this] def visitNode[T <: BaseIR](
    pre: T => T,
    transform: T => Option[T],
    post: T => T
  ): T => T = { t =>
    val t1 = pre(t)
    transform(t1) match {
      case Some(t2) => post(t2)
      case None => t1
    }
  }

  private[this] def simplifyValue: IR => IR =
    visitNode(
      simplifyValueChildren,
      rewriteValueNode,
      simplifyValue)

  private[this] def simplifyTable(allowRepartitioning: Boolean): TableIR => TableIR =
    visitNode(
      simplifyTableChildren(allowRepartitioning),
      rewriteTableNode(allowRepartitioning),
      simplifyTable(allowRepartitioning))

  private[this] def simplifyMatrix(allowRepartitioning: Boolean): MatrixIR => MatrixIR =
    visitNode(
      simplifyMatrixChildren(allowRepartitioning),
      rewriteMatrixNode(allowRepartitioning),
      simplifyMatrix(allowRepartitioning))

  private[this] def simplifyValueChildren(ir: IR): IR = {
    val newChildren = ir.children.map(Simplify.apply)
    if ((ir.children, newChildren).zipped.forall(_ eq _))
      ir
    else
      ir.copy(newChildren)
  }

  private[this] def simplifyTableChildren(allowRepartitioning: Boolean)(tir: TableIR): TableIR =
    simplifyChildren(tir, allowRepartitioning).asInstanceOf[TableIR]

  private[this] def simplifyMatrixChildren(allowRepartitioning: Boolean)(mir: MatrixIR): MatrixIR =
    simplifyChildren(mir, allowRepartitioning).asInstanceOf[MatrixIR]

  private[this] def simplifyChildren(ir: BaseIR, allowRepartitioning: Boolean): BaseIR = {
    val newChildren = ir.children.map(Simplify(_, allowRepartitioning && isDeterministic(ir)))

    if ((ir.children, newChildren).zipped.forall(_ eq _))
      ir
    else
      ir.copy(newChildren)
  }

  private[this] def rewriteValueNode: IR => Option[IR] = valueRules.lift

  private[this] def rewriteTableNode(allowRepartitioning: Boolean)(tir: TableIR): Option[TableIR] =
    tableRules(allowRepartitioning && isDeterministic(tir)).lift(tir)

  private[this] def rewriteMatrixNode(allowRepartitioning: Boolean)(mir: MatrixIR): Option[MatrixIR] =
    matrixRules(allowRepartitioning && isDeterministic(mir)).lift(mir)

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

  /** Returns true if 'ir' all value IR children of 'ir' are deterministic.
    */
  private[this] def isDeterministic(ir: BaseIR): Boolean = {
    ir.children.forall{
      case child: IR => !Exists(child, _.isInstanceOf[ApplySeeded])
      case _ => true
    }
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

    case x@ArrayMap(NA(_), _, _) => NA(x.typ)

    case x@ArrayFlatMap(NA(_), _, _) => NA(x.typ)

    case x@ArrayFilter(NA(_), _, _) => NA(x.typ)

    case x@ArrayFold(NA(_), _, _, _, _) => NA(x.typ)

    case IsNA(NA(_)) => True()

    case IsNA(x) if isDefinitelyDefined(x) => False()

    case Let(n1, v, Ref(n2, _)) if n1 == n2 => v

    case x@If(True(), cnsq, _) if x.typ == cnsq.typ => cnsq
    case x@If(False(), _, altr) if x.typ == altr.typ => altr

    case If(c, cnsq, altr) if cnsq == altr =>
      if (cnsq.typ.required)
        cnsq
      else
        If(IsNA(c), NA(cnsq.typ), cnsq)

    case Cast(x, t) if x.typ == t => x

    case ArrayLen(MakeArray(args, _)) => I32(args.length)

    case ArrayRef(MakeArray(args, _), I32(i)) if i >= 0 && i < args.length => args(i)

    case ArrayFilter(a, _, True()) => a

    case Let(n, v, b) if !Mentions(b, n) => b

    case Let(_, _, Begin(Seq())) => Begin(FastIndexedSeq())

    case ArrayFor(_, _, Begin(Seq())) => Begin(FastIndexedSeq())

    case ArrayFold(ArrayMap(a, n1, b), zero, accumName, valueName, body) => ArrayFold(a, zero, accumName, n1, Let(valueName, b, body))

    case ArrayLen(ArrayRange(start, end, I32(1))) => ApplyBinaryPrimOp(Subtract(), end, start)

    case ArrayLen(ArrayMap(a, _, _)) => ArrayLen(a)

    case ArrayLen(ArrayFlatMap(a, _, MakeArray(args, _))) => ApplyBinaryPrimOp(Multiply(), I32(args.length), ArrayLen(a))

    case ArrayFlatMap(ArrayMap(a, n1, b1), n2, b2) =>
      ArrayFlatMap(a, n1, Let(n2, b1, b2))

    case ArrayMap(ArrayMap(a, n1, b1), n2, b2) =>
      ArrayMap(a, n1, Let(n2, b1, b2))

    case GetField(MakeStruct(fields), name) =>
      val (_, x) = fields.find { case (n, _) => n == name }.get
      x

    case GetField(InsertFields(old, fields), name) =>
      fields.find { case (n, _) => n == name } match {
        case Some((_, x)) => x
        case None => GetField(old, name)
      }

    case InsertFields(InsertFields(base, fields1), fields2) =>
      val fields1Set = fields1.map(_._1).toSet
      val fields2Map = fields2.toMap

      val finalFields = fields1.map { case (name, fieldIR) => name -> fields2Map.getOrElse(name, fieldIR) } ++
        fields2.filter { case (name, _) => !fields1Set.contains(name) }
      InsertFields(base, finalFields)

    case InsertFields(MakeStruct(fields1), fields2) =>
      val fields1Set = fields1.map(_._1).toSet
      val fields2Map = fields2.toMap

      val finalFields = fields1.map { case (name, fieldIR) => name -> fields2Map.getOrElse(name, fieldIR) } ++
        fields2.filter { case (name, _) => !fields1Set.contains(name) }
      MakeStruct(finalFields)

    case InsertFields(struct, Seq()) => struct

    case GetField(SelectFields(old, fields), name) => GetField(old, name)

    case SelectFields(MakeStruct(fields), fieldNames) =>
      val makeStructFields = fields.toMap
      MakeStruct(fieldNames.map(f => f -> makeStructFields(f)))

    case x@SelectFields(InsertFields(struct, insertFields), selectFields) =>
      val selectSet = selectFields.toSet
      val insertFields2 = insertFields.filter { case (fName, _) => selectSet.contains(fName) }
      val structSet = struct.typ.asInstanceOf[TStruct].fieldNames.toSet
      val selectFields2 = selectFields.filter(structSet.contains)
      val x2 = InsertFields(SelectFields(struct, selectFields2), insertFields2)
      assert(x2.typ == x.typ)
      x2

    case GetTupleElement(MakeTuple(xs), idx) => xs(idx)

    case ApplyIR("annotate", Seq(s1, MakeStruct(Seq())), _) =>
      s1

    case TableCount(TableMapGlobals(child, _)) => TableCount(child)

    case TableCount(TableMapRows(child, _)) => TableCount(child)

    case TableCount(TableRepartition(child, _, _)) => TableCount(child)

    case TableCount(TableUnion(children)) =>
      children.map(TableCount).reduce[IR](ApplyBinaryPrimOp(Add(), _, _))

    case TableCount(TableKeyBy(child, _, _)) => TableCount(child)

    case TableCount(TableOrderBy(child, _)) => TableCount(child)

    case TableCount(TableLeftJoinRightDistinct(child, _, _)) => TableCount(child)

    case TableCount(TableRange(n, _)) => I64(n)

    case TableCount(TableParallelize(rows, _)) => Cast(ArrayLen(rows), TInt64())

    case TableCount(TableRename(child, _, _)) => TableCount(child)

    case TableCount(TableAggregateByKey(child, _)) => TableCount(TableDistinct(child))

    case ApplyIR("annotate", Seq(s, MakeStruct(fields)), _) =>
      InsertFields(s, fields)

    case SelectFields(SelectFields(old, _), fields) =>
      SelectFields(old, fields)
  }

  private[this] def tableRules(canRepartition: Boolean): PartialFunction[TableIR, TableIR] = {
    case TableFilter(t, True()) => t

    case TableFilter(TableRead(path, spec, typ, _), False() | NA(_)) =>
      TableRead(path, spec, typ, dropRows = true)

    case TableFilter(TableFilter(t, p1), p2) =>
      TableFilter(t,
        ApplySpecial("&&", Array(p1, p2)))

    case TableFilter(TableOrderBy(child, sortFields), pred) if canRepartition =>
      TableOrderBy(TableFilter(child, pred), sortFields)

    case TableKeyBy(TableOrderBy(child, sortFields), keys, false) if canRepartition =>
      TableKeyBy(child, keys, false)

    case TableKeyBy(TableKeyBy(child, _, _), keys, false) if canRepartition =>
      TableKeyBy(child, keys, false)

    case TableKeyBy(TableKeyBy(child, _, true), keys, true) if canRepartition =>
      TableKeyBy(child, keys, true)

    case TableKeyBy(child, key, _) if key == child.typ.key => child

    // TODO: Write more rules like this to bubble 'TableRename' nodes towards the root.
    case t@TableRename(TableKeyBy(child, keys, isSorted), rowMap, globalMap) =>
      TableKeyBy(TableRename(child, rowMap, globalMap), keys.map(t.rowF), isSorted)

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

    case TableMapGlobals(child, Ref("global", _)) => child

    case TableRename(child, m1, m2) if m1.isEmpty && m2.isEmpty => child

    // flatten unions
    case TableUnion(children) if children.exists(_.isInstanceOf[TableUnion]) & canRepartition =>
      TableUnion(children.flatMap {
        case u: TableUnion => u.children
        case c => Some(c)
      })

    // FIXME: currently doesn't work because TableUnion makes no guarantee on order. Put back in once ordering is enforced on Tables
    //      case MatrixRowsTable(MatrixUnionRows(children)) =>
    //        TableUnion(children.map(MatrixRowsTable))

    case MatrixColsTable(MatrixUnionRows(children)) =>
      MatrixColsTable(children(0))

    // Ignore column or row data that is immediately dropped
    case MatrixRowsTable(MatrixRead(typ, false, dropRows, reader)) =>
      MatrixRowsTable(MatrixRead(typ, dropCols = true, dropRows, reader))

    case MatrixColsTable(MatrixRead(typ, dropCols, false, reader)) =>
      MatrixColsTable(MatrixRead(typ, dropCols, dropRows = true, reader))

    case MatrixRowsTable(MatrixFilterRows(child, newRow))
      if !Mentions(newRow, "g") && !Mentions(newRow, "sa") && !ContainsAgg(newRow) =>
      val mrt = MatrixRowsTable(child)
      TableFilter(
        mrt,
        Subst(newRow, Env.empty[IR].bind("va" -> Ref("row", mrt.typ.rowType))))
    case MatrixRowsTable(MatrixMapGlobals(child, newRow)) => TableMapGlobals(MatrixRowsTable(child), newRow)
    case MatrixRowsTable(MatrixMapCols(child, _, _)) => MatrixRowsTable(child)
    case MatrixRowsTable(MatrixMapEntries(child, _)) => MatrixRowsTable(child)
    case MatrixRowsTable(MatrixFilterEntries(child, _)) => MatrixRowsTable(child)
    case MatrixRowsTable(MatrixFilterCols(child, _)) => MatrixRowsTable(child)
    case MatrixRowsTable(MatrixAggregateColsByKey(child, _, _)) => MatrixRowsTable(child)
    case MatrixRowsTable(MatrixChooseCols(child, _)) => MatrixRowsTable(child)
    case MatrixRowsTable(MatrixCollectColsByKey(child)) => MatrixRowsTable(child)
    case MatrixRowsTable(MatrixKeyRowsBy(child, keys, isSorted)) => TableKeyBy(MatrixRowsTable(child), keys, isSorted)

    case MatrixColsTable(x@MatrixMapCols(child, newRow, newKey))
      if newKey.isEmpty && !Mentions(newRow, "g") && !Mentions(newRow, "va") &&
        !ContainsAgg(newRow) && canRepartition && isDeterministic(x) && !ContainsScan(newRow) =>
      val mct = MatrixColsTable(child)
      TableMapRows(
        mct,
        Subst(newRow, Env.empty[IR].bind("sa" -> Ref("row", mct.typ.rowType))))
    case MatrixColsTable(MatrixFilterCols(child, pred))
      if !Mentions(pred, "g") && !Mentions(pred, "va") =>
      val mct = MatrixColsTable(child)
      TableFilter(
        mct,
        Subst(pred, Env.empty[IR].bind("sa" -> Ref("row", mct.typ.rowType))))
    case MatrixColsTable(MatrixMapGlobals(child, newRow)) => TableMapGlobals(MatrixColsTable(child), newRow)
    case MatrixColsTable(MatrixMapRows(child, _)) => MatrixColsTable(child)
    case MatrixColsTable(MatrixMapEntries(child, _)) => MatrixColsTable(child)
    case MatrixColsTable(MatrixFilterEntries(child, _)) => MatrixColsTable(child)
    case MatrixColsTable(MatrixFilterRows(child, _)) => MatrixColsTable(child)
    case MatrixColsTable(MatrixAggregateRowsByKey(child, _, _)) => MatrixColsTable(child)
    case MatrixColsTable(MatrixKeyRowsBy(child, _, _)) => MatrixColsTable(child)

    case TableHead(TableMapRows(child, newRow), n) =>
      TableMapRows(TableHead(child, n), newRow)
    case TableHead(TableRepartition(child, nPar, shuffle), n) if canRepartition =>
      TableRepartition(TableHead(child, n), nPar, shuffle)
    case TableHead(tr@TableRange(nRows, nPar), n) if canRepartition =>
      if (n < nRows)
        TableRange(n.toInt, (nPar.toFloat * n / nRows).toInt.max(1))
      else
        tr
    case TableHead(TableMapGlobals(child, newRow), n) =>
      TableMapGlobals(TableHead(child, n), newRow)
    case TableHead(TableOrderBy(child, sortFields), n)
      if sortFields.forall(_.sortOrder == Ascending) && n < 256 && canRepartition =>
      // n < 256 is arbitrary for memory concerns
      val uid = genUID()
      val row = Ref("row", child.typ.rowType)
      val keyStruct = MakeStruct(sortFields.map(f => f.field -> GetField(row, f.field)))
      val aggSig = AggSignature(TakeBy(), FastSeq(TInt32()), None, FastSeq(row.typ, keyStruct.typ))
      val te =
        TableExplode(
          TableKeyByAndAggregate(child,
            MakeStruct(Seq(
              "row" -> ApplyAggOp(
                Array(row, keyStruct),
                FastIndexedSeq(I32(n.toInt)),
                None,
                aggSig))),
            MakeStruct(Seq()), // aggregate to one row
            Some(1), 10),
          "row")
      TableMapRows(te, GetField(Ref("row", te.typ.rowType), "row"))

    case TableKeyByAndAggregate(child, MakeStruct(Seq()), k@MakeStruct(keyFields), _, _) if canRepartition =>
      TableDistinct(TableMapRows(child, k))
    case TableKeyByAndAggregate(child, expr, newKey, _, _)
      if newKey == MakeStruct(child.typ.key.map(k => k -> GetField(Ref("row", child.typ.rowType), k)))
        && child.typ.key.nonEmpty && canRepartition =>
      TableAggregateByKey(child, expr)
    case TableAggregateByKey(TableKeyBy(child, keys, _), expr) if canRepartition =>
      TableKeyByAndAggregate(child, expr, MakeStruct(keys.map(k => k -> GetField(Ref("row", child.typ.rowType), k))))
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

    case MatrixFilterRows(MatrixFilterRows(child, pred1), pred2) => MatrixFilterRows(child, ApplySpecial("&&", FastSeq(pred1, pred2)))

    case MatrixFilterCols(MatrixFilterCols(child, pred1), pred2) => MatrixFilterCols(child, ApplySpecial("&&", FastSeq(pred1, pred2)))
      
    case MatrixFilterEntries(MatrixFilterEntries(child, pred1), pred2) => MatrixFilterEntries(child, ApplySpecial("&&", FastSeq(pred1, pred2)))
  }
}

package is.hail.expr.ir

import is.hail.utils._
import is.hail.expr._

object Simplify {
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
      case _ => false
    }
  }

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

  def apply(ir: BaseIR): BaseIR = {
    RewriteBottomUp(ir, matchErrorToNone {
      // optimize IR

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

      case If(True(), x, _) => x
      case If(False(), _, x) => x

      case If(c, cnsq, altr) if cnsq == altr =>
        If(IsNA(c), NA(cnsq.typ), cnsq)

      case Cast(x, t) if x.typ == t => x

      case ArrayLen(MakeArray(args, _)) => I32(args.length)

      case ArrayRef(MakeArray(args, _), I32(i)) if i >= 0 && i < args.length => args(i)

      case ArrayFilter(a, _, True()) => a

      case Let(n, v, b) if !Mentions(b, n) => b

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

      case GetField(SelectFields(old, fields), name) => GetField(old, name)

      case SelectFields(MakeStruct(fields), fieldNames) =>
        val makeStructFields = fields.toMap
        MakeStruct(fieldNames.map(f => f -> makeStructFields(f)))

      case GetTupleElement(MakeTuple(xs), idx) => xs(idx)

      // optimize TableIR
      case TableFilter(t, True()) => t

      case TableFilter(TableRead(path, spec, typ, _), False() | NA(_)) =>
        TableRead(path, spec, typ, dropRows = true)

      case TableFilter(TableFilter(t, p1), p2) =>
        TableFilter(t,
          ApplySpecial("&&", Array(p1, p2)))

      case TableFilter(TableOrderBy(child, sortFields), pred) =>
        TableOrderBy(TableFilter(child, pred), sortFields)

      case TableKeyBy(TableOrderBy(child, sortFields), keys, nPartitionKeys, true) =>
        TableKeyBy(child, keys, nPartitionKeys, true)

      case TableCount(TableMapGlobals(child, _, _)) => TableCount(child)

      case TableCount(TableMapRows(child, _, _, _)) => TableCount(child)

      case TableCount(TableUnion(children)) =>
        children.map(TableCount).reduce[IR](ApplyBinaryPrimOp(Add(), _, _))

      case TableCount(TableKeyBy(child, _, _, _)) => TableCount(child)

      case TableCount(TableOrderBy(child, _)) => TableCount(child)

      case TableCount(TableUnkey(child)) => TableCount(child)

      case TableCount(TableRange(n, _)) => I64(n)

      case TableCount(TableParallelize(_, rows, _)) => I64(rows.length)

        // flatten unions
      case TableUnion(children) if children.exists(_.isInstanceOf[TableUnion]) =>
        TableUnion(children.flatMap {
          case u: TableUnion => u.children
          case c => Some(c)
        })

      // optimize MatrixIR

      // Equivalent rewrites for the new Filter{Cols,Rows}IR
      case MatrixFilterRows(MatrixRead(typ, partitionCounts, colCount, dropCols, _, reader), False() | NA(_)) =>
        MatrixRead(typ, Some(FastIndexedSeq()), colCount, dropCols, dropRows = true, reader)

      case MatrixFilterCols(MatrixRead(typ, partitionCounts, _, _, dropRows, reader), False() | NA(_)) =>
        MatrixRead(typ, partitionCounts, Some(0), dropCols = true, dropRows, reader)

      // Ignore column or row data that is immediately dropped
      case MatrixRowsTable(MatrixRead(typ, partitionCounts, _, false, dropRows, reader)) =>
        MatrixRowsTable(MatrixRead(typ, partitionCounts, Some(0), dropCols = true, dropRows, reader))

      case MatrixColsTable(MatrixRead(typ, partitionCounts, colCount, dropCols, false, reader)) =>
        MatrixColsTable(MatrixRead(typ, Some(FastIndexedSeq()), colCount, dropCols, dropRows = true, reader))

      // Keep all rows/cols = do nothing
      case MatrixFilterRows(m, True()) => m

      case MatrixFilterCols(m, True()) => m

      case MatrixRowsTable(MatrixFilterRows(child, newRow))
        if !Mentions(newRow, "g") && !Mentions(newRow, "sa") && !ContainsAgg(newRow) =>
        val mrt = MatrixRowsTable(child)
        TableFilter(
          mrt,
          Subst(newRow, Env.empty[IR].bind("va" -> Ref("row", mrt.typ.rowType))))
      case MatrixRowsTable(MatrixMapGlobals(child, newRow, value)) => TableMapGlobals(MatrixRowsTable(child), newRow, value)
      case MatrixRowsTable(MatrixMapCols(child, _, _)) => MatrixRowsTable(child)
      case MatrixRowsTable(MatrixMapEntries(child, _)) => MatrixRowsTable(child)
      case MatrixRowsTable(MatrixFilterEntries(child, _)) => MatrixRowsTable(child)
      case MatrixRowsTable(MatrixFilterCols(child, _)) => MatrixRowsTable(child)
      case MatrixRowsTable(MatrixAggregateColsByKey(child, _)) => MatrixRowsTable(child)
      case MatrixRowsTable(MatrixChooseCols(child, _)) => MatrixRowsTable(child)
      case MatrixRowsTable(MatrixCollectColsByKey(child)) => MatrixRowsTable(child)

      case MatrixColsTable(MatrixMapCols(child, newRow, newKey))
        if newKey.isEmpty && !Mentions(newRow, "g") && !Mentions(newRow, "va") && !ContainsAgg(newRow) =>
        val mct = MatrixColsTable(child)
        TableMapRows(
          mct,
          Subst(newRow, Env.empty[IR].bind("sa" -> Ref("row", mct.typ.rowType))),
          Some(child.typ.colKey),
          Some(child.typ.colKey.length))
      case MatrixColsTable(MatrixFilterCols(child, newRow))
        if !Mentions(newRow, "g") && !Mentions(newRow, "va") && !ContainsAgg(newRow) =>
        val mct = MatrixColsTable(child)
        TableFilter(
          mct,
          Subst(newRow, Env.empty[IR].bind("sa" -> Ref("row", mct.typ.rowType))))
      case MatrixColsTable(MatrixMapGlobals(child, newRow, value)) => TableMapGlobals(MatrixColsTable(child), newRow, value)
      case MatrixColsTable(MatrixMapRows(child, _, _)) => MatrixColsTable(child)
      case MatrixColsTable(MatrixMapEntries(child, _)) => MatrixColsTable(child)
      case MatrixColsTable(MatrixFilterEntries(child, _)) => MatrixColsTable(child)
      case MatrixColsTable(MatrixFilterRows(child, _)) => MatrixColsTable(child)
      case MatrixColsTable(MatrixAggregateRowsByKey(child, _)) => MatrixColsTable(child)

      case TableCount(TableAggregateByKey(child, _)) => TableCount(TableDistinct(child))
    })
  }
}

package is.hail.expr.ir.analyses

import is.hail.expr.ir._
import is.hail.utils.{toRichOption, FastSeq, PartitionCounts}

object RowCounts {
  object PartitionCounts {
    def unapply(ir: BaseIR): Option[IndexedSeq[Long]] = partitionCounts(ir)
  }

  def partitionCounts(ir: BaseIR): Option[IndexedSeq[Long]] = ir match {
    case ir: TableRead =>
      if (ir.dropRows) Some(FastSeq(0L)) else ir.tr.partitionCounts
    case ir: MatrixRead =>
      if (ir.dropRows) Some(FastSeq.empty) else ir.reader.partitionCounts
    case range: TableRange => Some(range.partitionCounts.map(_.toLong))
    case TableHead(child, n) =>
      partitionCounts(child).map(PartitionCounts.getHeadPCs(_, n))
    case TableTail(child, n) =>
      partitionCounts(child).map(PartitionCounts.getTailPCs(_, n))
    case MatrixRowsHead(child, n) =>
      partitionCounts(child).map(PartitionCounts.getHeadPCs(_, n))
    case MatrixRowsTail(child, n) =>
      partitionCounts(child).map(PartitionCounts.getTailPCs(_, n))
    case PreservesRows(child, true) => partitionCounts(child)
    case _ => None
  }

  def rowCountUpperBound(ir: BaseIR): Option[Long] = ir match {
    case ir: TableRead => partitionCounts(ir).map(_.sum)
    case ir: MatrixRead => partitionCounts(ir).map(_.sum)
    case TableRange(n, _) => Some(n.toLong)
    case TableHead(child, n) => Some(rowCountUpperBound(child).getOrElse(Long.MaxValue).min(n))
    case TableTail(child, n) => Some(rowCountUpperBound(child).getOrElse(Long.MaxValue).min(n))
    case MatrixRowsHead(child, n) => Some(rowCountUpperBound(child).getOrElse(Long.MaxValue).min(n))
    case MatrixRowsTail(child, n) => Some(rowCountUpperBound(child).getOrElse(Long.MaxValue).min(n))
    case TableUnion(children) =>
      children.foldLeft[Option[Long]](Some(0)) { (sum, child) =>
        rowCountUpperBound(child).liftedZip(sum).map { case (l, r) => l + r }
      }
    case MatrixUnionRows(children) =>
      children.foldLeft[Option[Long]](Some(0)) { (sum, child) =>
        rowCountUpperBound(child).liftedZip(sum).map { case (l, r) => l + r }
      }
    case MatrixUnionCols(left, right, "inner") =>
      (rowCountUpperBound(left), rowCountUpperBound(right)) match {
        case (None, None) => None
        case (l, r) => Some(Math.min(l.getOrElse(Long.MaxValue), r.getOrElse(Long.MaxValue)))
      }
    case MatrixUnionCols(left, right, _) =>
      rowCountUpperBound(left).liftedZip(rowCountUpperBound(right)).map { case (l, r) => l + r }
    case PreservesOrRemovesRows(child) => rowCountUpperBound(child)
    case _ => None
  }
}

object ColumnCount {
  def unapply(ir: MatrixIR): Option[Int] = apply(ir)

  def apply(ir: MatrixIR): Option[Int] = ir match {
    case ir: MatrixRead => if (ir.dropCols) Some(0) else ir.reader.columnCount
    case MatrixChooseCols(_, oldIndices) => Some(oldIndices.length)
    case MatrixColsHead(child, n) => ColumnCount(child).map(n.min)
    case MatrixColsTail(child, n) => ColumnCount(child).map(n.min)
    case MatrixUnionRows(children) =>
      children.foldLeft[Option[Int]](None) { (acc, child) =>
        val count = ColumnCount(child)
        acc.liftedZip(count).foreach { case (l, r) => assert(l == r) }
        acc.orElse(count)
      }
    case MatrixUnionCols(left, right, _) =>
      ColumnCount(left).liftedZip(ColumnCount(right)).map { case (l, r) => l + r }
    case PreservesCols(child: MatrixIR) => ColumnCount(child)
    case _ => None
  }
}

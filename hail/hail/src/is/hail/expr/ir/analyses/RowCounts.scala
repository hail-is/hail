package is.hail.expr.ir.analyses

import is.hail.collection.FastSeq
import is.hail.collection.implicits.toRichOption
import is.hail.expr.ir._
import is.hail.utils.PartitionCounts.{getHeadPCs, getTailPCs}

object PartitionCounts {
  def unapply(ir: BaseIR): Option[IndexedSeq[Long]] = PartitionCounts(ir)

  def apply(ir: BaseIR): Option[IndexedSeq[Long]] = ir match {
    case ir: TableRead =>
      if (ir.dropRows) Some(FastSeq(0L)) else ir.tr.partitionCounts
    case ir: MatrixRead =>
      if (ir.dropRows) Some(FastSeq.empty) else ir.reader.partitionCounts
    case range: TableRange => Some(range.partitionCounts.map(_.toLong))
    case TableHead(child, n) =>
      PartitionCounts(child).map(getHeadPCs(_, n))
    case TableTail(child, n) =>
      PartitionCounts(child).map(getTailPCs(_, n))
    case MatrixRowsHead(child, n) =>
      PartitionCounts(child).map(getHeadPCs(_, n))
    case MatrixRowsTail(child, n) =>
      PartitionCounts(child).map(getTailPCs(_, n))
    case PreservesRows(child, true) => PartitionCounts(child)
    case _ => None
  }
}

object RowCountUpperBound {
  def apply(ir: BaseIR): Option[Long] = ir match {
    case ir: TableRead => PartitionCounts(ir).map(_.sum)
    case ir: MatrixRead => PartitionCounts(ir).map(_.sum)
    case TableRange(n, _) => Some(n.toLong)
    case TableHead(child, n) => Some(RowCountUpperBound(child).getOrElse(Long.MaxValue).min(n))
    case TableTail(child, n) => Some(RowCountUpperBound(child).getOrElse(Long.MaxValue).min(n))
    case MatrixRowsHead(child, n) => Some(RowCountUpperBound(child).getOrElse(Long.MaxValue).min(n))
    case MatrixRowsTail(child, n) => Some(RowCountUpperBound(child).getOrElse(Long.MaxValue).min(n))
    case TableUnion(children) =>
      children.foldLeft[Option[Long]](Some(0)) { (sum, child) =>
        RowCountUpperBound(child).liftedZip(sum).map { case (l, r) => l + r }
      }
    case MatrixUnionRows(children) =>
      children.foldLeft[Option[Long]](Some(0)) { (sum, child) =>
        RowCountUpperBound(child).liftedZip(sum).map { case (l, r) => l + r }
      }
    case MatrixUnionCols(left, right, "inner") =>
      (RowCountUpperBound(left), RowCountUpperBound(right)) match {
        case (None, None) => None
        case (l, r) => Some(Math.min(l.getOrElse(Long.MaxValue), r.getOrElse(Long.MaxValue)))
      }
    case MatrixUnionCols(left, right, _) =>
      RowCountUpperBound(left).liftedZip(RowCountUpperBound(right)).map { case (l, r) => l + r }
    case PreservesOrRemovesRows(child) => RowCountUpperBound(child)
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

object PartitionCountsOrColumnCount {
  def unapply(ir: MatrixIR): Option[(Option[IndexedSeq[Long]], Option[Int])] =
    (PartitionCounts(ir), ColumnCount(ir)) match {
      case (None, None) => None
      case x => Some(x)
    }
}

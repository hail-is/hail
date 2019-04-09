package is.hail.methods

import is.hail.expr.ir.{MatrixToMatrixApply, MatrixValue, TableToTableApply, TableValue}
import is.hail.expr.ir.functions.{MatrixToMatrixFunction, TableToTableFunction}
import is.hail.expr.types.{MatrixType, TableType}
import is.hail.rvd.RVDType

case class TableFilterPartitions(parts: IndexedSeq[Int], keep: Boolean) extends TableToTableFunction {
  override def preservesPartitionCounts: Boolean = false

  override def typeInfo(childType: TableType, childRVDType: RVDType): (TableType, RVDType) = (childType, childRVDType)

  override def execute(tv: TableValue): TableValue = {
    val newRVD = if (keep)
      tv.rvd.subsetPartitions(parts.toArray)
    else {
      val subtract = parts.toSet
      tv.rvd.subsetPartitions((0 until tv.rvd.getNumPartitions).filter(i => !subtract.contains(i)).toArray)
    }
    tv.copy(rvd = newRVD)
  }
}

case class MatrixFilterPartitions(parts: IndexedSeq[Int], keep: Boolean) extends MatrixToMatrixFunction {
  override def preservesPartitionCounts: Boolean = false

  override def typeInfo(childType: MatrixType, childRVDType: RVDType): (MatrixType, RVDType) = (childType, childRVDType)

  override def execute(mv: MatrixValue): MatrixValue = throw new UnsupportedOperationException

  override def lower(): Option[TableToTableFunction] = Some(TableFilterPartitions(parts, keep))
}

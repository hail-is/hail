package is.hail.methods

import is.hail.expr.ir.{MatrixToMatrixApply, MatrixValue, TableToTableApply, TableValue}
import is.hail.expr.ir.functions.{MatrixToMatrixFunction, TableToTableFunction}
import is.hail.expr.types.{MatrixType, TableType}
import is.hail.rvd.RVDType

class TableFilterPartitions(parts: Array[Int], keep: Boolean) extends TableToTableFunction {
  override def preservesPartitionCounts: Boolean = false

  override def typeInfo(childType: TableType, childRVDType: RVDType): (TableType, RVDType) = (childType, childRVDType)

  override def execute(tv: TableValue): TableValue = {
    val newRVD = if (keep)
      tv.rvd.subsetPartitions(parts)
    else {
      val subtract = parts.toSet
      tv.rvd.subsetPartitions((0 until tv.rvd.getNumPartitions).filter(i => !subtract.contains(i)).toArray)
    }
    tv.copy(rvd = newRVD)
  }
}

class MatrixFilterPartitions(parts: Array[Int], keep: Boolean) extends MatrixToMatrixFunction {
  override def preservesPartitionCounts: Boolean = false

  override def typeInfo(childType: MatrixType, childRVDType: RVDType): (MatrixType, RVDType) = (childType, childRVDType)

  override def execute(mv: MatrixValue): MatrixValue = {
    val newRVD = if (keep)
      mv.rvd.subsetPartitions(parts)
    else {
      val subtract = parts.toSet
      mv.rvd.subsetPartitions((0 until mv.rvd.getNumPartitions).filter(i => !subtract.contains(i)).toArray)
    }
    mv.copy(rvd = newRVD)
  }
}

package is.hail.methods

import is.hail.expr.ir.TableValue
import is.hail.expr.ir.functions.{MatrixToMatrixFunction, TableToTableFunction}
import is.hail.expr.ir.lowering.MonadLower
import is.hail.types.{MatrixType, TableType}

import scala.language.higherKinds

case class TableFilterPartitions(parts: Seq[Int], keep: Boolean) extends TableToTableFunction {
  override def preservesPartitionCounts: Boolean = false

  override def typ(childType: TableType): TableType = childType

  override def execute[M[_]](tv: TableValue)(implicit M: MonadLower[M]): M[TableValue] =
    M.pure {
      val newRVD = if (keep)
        tv.rvd.subsetPartitions(parts.toArray)
      else {
        val subtract = parts.toSet
        tv.rvd.subsetPartitions((0 until tv.rvd.getNumPartitions).filter(i => !subtract.contains(i)).toArray)
      }
      tv.copy(rvd = newRVD)
    }
}

case class MatrixFilterPartitions(parts: Seq[Int], keep: Boolean) extends MatrixToMatrixFunction {
  override def preservesPartitionCounts: Boolean = false

  override def typ(childType: MatrixType): MatrixType = childType

  override def lower(): TableToTableFunction = TableFilterPartitions(parts, keep)
}

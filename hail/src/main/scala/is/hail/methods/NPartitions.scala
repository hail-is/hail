package is.hail.methods

import is.hail.expr.ir.functions.{MatrixToValueFunction, TableToValueFunction}
import is.hail.expr.ir.{ExecuteContext, MatrixValue, TableValue}
import is.hail.expr.types.virtual.{TInt32, Type}
import is.hail.expr.types.{MatrixType, TableType}

case class NPartitionsTable() extends TableToValueFunction {
  override def typ(childType: TableType): Type = TInt32

  override def execute(ctx: ExecuteContext, tv: TableValue): Any = tv.rvd.getNumPartitions
}

case class NPartitionsMatrixTable() extends MatrixToValueFunction {
  override def typ(childType: MatrixType): Type = TInt32

  override def execute(ctx: ExecuteContext, mv: MatrixValue): Any = mv.rvd.getNumPartitions

  override def lower(): Option[TableToValueFunction] = Some(NPartitionsTable())
}

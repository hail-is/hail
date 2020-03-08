package is.hail.methods

import is.hail.expr.ir.functions.{MatrixToValueFunction, TableToValueFunction}
import is.hail.expr.ir.{ExecuteContext, MatrixValue, TableValue}
import is.hail.expr.types.virtual.{TInt64, Type}
import is.hail.expr.types.{MatrixType, TableType}

case class ForceCountTable() extends TableToValueFunction {
  override def typ(childType: TableType): Type = TInt64

  override def execute(ctx: ExecuteContext, tv: TableValue): Any = tv.rvd.count()
}

case class ForceCountMatrixTable() extends MatrixToValueFunction {
  override def typ(childType: MatrixType): Type = TInt64

  override def execute(ctx: ExecuteContext, mv: MatrixValue): Any = throw new UnsupportedOperationException

  override def lower(): Option[TableToValueFunction] = Some(ForceCountTable())
}

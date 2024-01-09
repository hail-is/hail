package is.hail.methods

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{MatrixValue, TableValue}
import is.hail.expr.ir.functions.{MatrixToValueFunction, TableToValueFunction}
import is.hail.types.{MatrixType, RTable, TableType, TypeWithRequiredness}
import is.hail.types.virtual.{TInt64, Type}

case class ForceCountTable() extends TableToValueFunction {
  override def typ(childType: TableType): Type = TInt64

  def unionRequiredness(childType: RTable, resultType: TypeWithRequiredness): Unit = ()

  override def execute(ctx: ExecuteContext, tv: TableValue): Any = tv.rvd.count()
}

case class ForceCountMatrixTable() extends MatrixToValueFunction {
  override def typ(childType: MatrixType): Type = TInt64

  def unionRequiredness(childType: RTable, resultType: TypeWithRequiredness): Unit = ()

  override def execute(ctx: ExecuteContext, mv: MatrixValue): Any =
    throw new UnsupportedOperationException

  override def lower(): Option[TableToValueFunction] = Some(ForceCountTable())
}

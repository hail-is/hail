package is.hail.methods

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{MatrixValue, TableValue}
import is.hail.expr.ir.functions.{MatrixToValueFunction, TableToValueFunction}
import is.hail.types.{RTable, TypeWithRequiredness}
import is.hail.types.virtual.{MatrixType, TInt32, TableType, Type}

case class NPartitionsTable() extends TableToValueFunction {
  override def typ(childType: TableType): Type = TInt32

  override def unionRequiredness(childType: RTable, resultType: TypeWithRequiredness): Unit = ()

  override def execute(ctx: ExecuteContext, tv: TableValue): Any = tv.rvd.getNumPartitions
}

case class NPartitionsMatrixTable() extends MatrixToValueFunction {
  override def typ(childType: MatrixType): Type = TInt32

  override def unionRequiredness(childType: RTable, resultType: TypeWithRequiredness): Unit = ()

  override def execute(ctx: ExecuteContext, mv: MatrixValue): Any = mv.rvd.getNumPartitions

  override def lower(): Option[TableToValueFunction] = Some(NPartitionsTable())
}

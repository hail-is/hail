package is.hail.methods

import is.hail.expr.ir.{MatrixValue, TableValue}
import is.hail.expr.ir.functions.{MatrixToValueFunction, TableToValueFunction}
import is.hail.expr.types.{MatrixType, TableType}
import is.hail.expr.types.virtual.{TInt64, Type}

case class ForceCountTable() extends TableToValueFunction {
  override def typ(childType: TableType): Type = TInt64()

  override def execute(tv: TableValue): Any = tv.rvd.count()
}

case class ForceCountMatrixTable() extends MatrixToValueFunction {
  override def typ(childType: MatrixType): Type = TInt64()

  override def execute(mv: MatrixValue): Any = mv.rvd.count()
}

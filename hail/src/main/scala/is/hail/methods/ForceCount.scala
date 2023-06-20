package is.hail.methods

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.functions.{MatrixToValueFunction, TableToValueFunction}
import is.hail.expr.ir.lowering.MonadLower
import is.hail.expr.ir.{MatrixValue, TableValue}
import is.hail.types.virtual.{TInt64, Type}
import is.hail.types.{MatrixType, RTable, TableType, TypeWithRequiredness}

import scala.language.higherKinds

case class ForceCountTable() extends TableToValueFunction {
  override def typ(childType: TableType): Type = TInt64

  def unionRequiredness(childType: RTable, resultType: TypeWithRequiredness): Unit = ()

  override def execute[M[_]](tv: TableValue)(implicit M: MonadLower[M]): M[Any] =
    M.pure(tv.rvd.count())
}

case class ForceCountMatrixTable() extends MatrixToValueFunction {
  override def typ(childType: MatrixType): Type = TInt64

  def unionRequiredness(childType: RTable, resultType: TypeWithRequiredness): Unit = ()

  override def execute[M[_]](mv: MatrixValue)(implicit M: MonadLower[M]): M[Any] =
    M.raiseError(new UnsupportedOperationException)

  override def lower(): Option[TableToValueFunction] = Some(ForceCountTable())
}

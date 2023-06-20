package is.hail.methods

import is.hail.expr.ir.functions.{MatrixToValueFunction, TableToValueFunction}
import is.hail.expr.ir.lowering.MonadLower
import is.hail.expr.ir.{MatrixValue, TableValue}
import is.hail.types.virtual.{TInt32, Type}
import is.hail.types.{MatrixType, RTable, TableType, TypeWithRequiredness}

import scala.language.higherKinds

case class NPartitionsTable() extends TableToValueFunction {
  override def typ(childType: TableType): Type = TInt32

  def unionRequiredness(childType: RTable, resultType: TypeWithRequiredness): Unit = ()

  override def execute[M[_]](tv: TableValue)(implicit M: MonadLower[M]): M[Any] =
    M.pure(tv.rvd.getNumPartitions)
}

case class NPartitionsMatrixTable() extends MatrixToValueFunction {
  override def typ(childType: MatrixType): Type = TInt32

  def unionRequiredness(childType: RTable, resultType: TypeWithRequiredness): Unit = ()

  override def execute[M[_]](mv: MatrixValue)(implicit M: MonadLower[M]): M[Any] =
    M.pure(mv.rvd.getNumPartitions)

  override def lower(): Option[TableToValueFunction] = Some(NPartitionsTable())
}

package is.hail.expr.ir.functions

import is.hail.expr.types.BlockMatrixType
import is.hail.expr.types.virtual.Type
import is.hail.linalg.BlockMatrix

case class GetElement(row: Long, col: Long) extends BlockMatrixToValueFunction {
  override def typ(childType: BlockMatrixType): Type = childType.elementType

  override def execute(bm: BlockMatrix): Any = bm.getElement(row, col)
}

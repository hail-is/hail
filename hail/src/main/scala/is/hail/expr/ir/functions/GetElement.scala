package is.hail.expr.ir.functions

import is.hail.expr.ir.ExecuteContext
import is.hail.expr.types.BlockMatrixType
import is.hail.expr.types.virtual.Type
import is.hail.linalg.BlockMatrix

case class GetElement(index: Seq[Long]) extends BlockMatrixToValueFunction {
  assert(index.length == 2)

  override def typ(childType: BlockMatrixType): Type = childType.elementType

  override def execute(ctx: ExecuteContext, bm: BlockMatrix): Any = bm.getElement(index(0), index(1))
}

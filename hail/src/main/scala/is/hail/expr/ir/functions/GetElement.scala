package is.hail.expr.ir.functions

import is.hail.backend.ExecuteContext
import is.hail.linalg.BlockMatrix
import is.hail.types.BlockMatrixType
import is.hail.types.virtual.Type

case class GetElement(index: IndexedSeq[Long]) extends BlockMatrixToValueFunction {
  assert(index.length == 2)

  override def typ(childType: BlockMatrixType): Type = childType.elementType

  override def execute(ctx: ExecuteContext, bm: BlockMatrix): Any =
    bm.getElement(index(0), index(1))
}

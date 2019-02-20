package is.hail.expr.ir.functions

import is.hail.expr.types.BlockMatrixType
import is.hail.expr.types.virtual.Type
import is.hail.linalg.BlockMatrix

case class GetElement(index: Seq[Long]) extends BlockMatrixToValueFunction {
  assert(index.length == 2)

  override def typ(childType: BlockMatrixType): Type = childType.elementType

  override def execute(bm: BlockMatrix): Any = bm.getElement(index.head, index(1))
}

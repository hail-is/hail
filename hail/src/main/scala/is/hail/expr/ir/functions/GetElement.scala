package is.hail.expr.ir.functions

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.lowering.MonadLower
import is.hail.types.BlockMatrixType
import is.hail.types.virtual.Type
import is.hail.linalg.BlockMatrix

import scala.language.higherKinds

case class GetElement(index: IndexedSeq[Long]) extends BlockMatrixToValueFunction {
  assert(index.length == 2)

  override def typ(childType: BlockMatrixType): Type = childType.elementType

  override def execute[M[_]](bm: BlockMatrix)(implicit M: MonadLower[M]): M[Any] =
    M.pure(bm.getElement(index(0), index(1)))
}

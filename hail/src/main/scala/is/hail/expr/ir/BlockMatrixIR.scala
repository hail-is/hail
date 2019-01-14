package is.hail.expr.ir

import is.hail.HailContext
import is.hail.expr.types.{BaseType, BlockMatrixType}
import is.hail.linalg.BlockMatrix
import is.hail.utils.fatal

abstract sealed class BlockMatrixIR extends BaseIR {
  protected[ir] def execute(hc: HailContext): BlockMatrix =
    fatal("tried to execute unexecutable IR:\n" + Pretty(this))
}

case class BlockMatrixRead(path: String) extends BlockMatrixIR {
  override def typ: BaseType = BlockMatrixType()

  override def children: IndexedSeq[BaseIR] = Array.empty[BlockMatrixIR]

  override def copy(newChildren: IndexedSeq[BaseIR]): BlockMatrixRead = {
    assert(newChildren.isEmpty)
    BlockMatrixRead(path)
  }

  override protected[ir] def execute(hc: HailContext): BlockMatrix = {
    BlockMatrix.read(hc, path)
  }
}
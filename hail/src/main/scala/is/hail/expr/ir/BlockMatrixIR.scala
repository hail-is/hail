package is.hail.expr.ir

import is.hail.HailContext
import is.hail.expr.types.{BaseType, BlockMatrixType}
import is.hail.linalg.BlockMatrix
import is.hail.utils.fatal

abstract sealed class BlockMatrixIR extends BaseIR {
  def typ: BlockMatrixType

  protected[ir] def execute(hc: HailContext): BlockMatrix =
    fatal("tried to execute unexecutable IR:\n" + Pretty(this))
}

case class BlockMatrixRead(path: String) extends BlockMatrixIR {
  override def typ: BlockMatrixType = {
    val metadata = BlockMatrix.readMetadata(HailContext.get, path)
    BlockMatrixType(metadata.nRows, metadata.nCols, metadata.blockSize)
  }

  override def children: IndexedSeq[BaseIR] = Array.empty[BlockMatrixIR]

  override def copy(newChildren: IndexedSeq[BaseIR]): BlockMatrixRead = {
    assert(newChildren.isEmpty)
    BlockMatrixRead(path)
  }

  override protected[ir] def execute(hc: HailContext): BlockMatrix = {
    BlockMatrix.read(hc, path)
  }
}

case class BlockMatrixAdd(left: BlockMatrixIR, right: BlockMatrixIR) extends BlockMatrixIR {

  //ASSUMPTION: both matrices are the same dimensions
  //TODO Check about block size, currently taken from left in BlockMatrix.add
  override def typ: BlockMatrixType = left.typ

  override def children: IndexedSeq[BaseIR] = Array(left, right)

  override def copy(newChildren: IndexedSeq[BaseIR]): BaseIR = {
    assert(newChildren.length == 2)
    BlockMatrixAdd(newChildren(0).asInstanceOf[BlockMatrixIR], newChildren(1).asInstanceOf[BlockMatrixIR])
  }

  override protected[ir] def execute(hc: HailContext): BlockMatrix = {
    left.execute(hc).add(right.execute(hc))
  }
}

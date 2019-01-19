package is.hail.expr.ir

import is.hail.HailContext
import is.hail.expr.types.BlockMatrixType
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
    BlockMatrixType(IndexedSeq(metadata.nRows, metadata.nCols), metadata.blockSize, IndexedSeq(true, true))
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

class BlockMatrixLiteral(value: BlockMatrix) extends BlockMatrixIR {
  override def typ: BlockMatrixType = {
    BlockMatrixType(IndexedSeq(value.nRows, value.nCols), value.blockSize, IndexedSeq(true, true))
  }

  override def children: IndexedSeq[BaseIR] = Array.empty[BlockMatrixIR]

  override def copy(newChildren: IndexedSeq[BaseIR]): BaseIR = {
    assert(newChildren.isEmpty)
    new BlockMatrixLiteral(value)
  }

  override protected[ir] def execute(hc: HailContext): BlockMatrix = value
}

case class BlockMatrixAdd(left: BlockMatrixIR, right: BlockMatrixIR) extends BlockMatrixIR {

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

case class BlockMatrixElementWiseBinaryOp(
  left: BlockMatrixIR,
  right: BlockMatrixIR,
  applyBinOp: ApplyBinaryPrimOp) extends BlockMatrixIR {
  override def typ: BlockMatrixType = left.typ

  override def children: IndexedSeq[BaseIR] = Array(left, right)

  override def copy(newChildren: IndexedSeq[BaseIR]): BaseIR = {
    assert(newChildren.length == 2)
    BlockMatrixElementWiseBinaryOp(
      newChildren(0).asInstanceOf[BlockMatrixIR],
      newChildren(1).asInstanceOf[BlockMatrixIR],
      applyBinOp)
  }

  override protected[ir] def execute(hc: HailContext): BlockMatrix = {
    val leftMatrix = left.execute(hc)
    val rightMatrix = right.execute(hc)

    (applyBinOp.l, applyBinOp.r, applyBinOp.op) match {
      case (Ref(_, _), Ref(_, _), Add()) => leftMatrix.add(rightMatrix)
      case (Ref(_, _), Ref(_, _), Multiply()) => leftMatrix.mul(rightMatrix)
      case (Ref("left", _), Ref("right", _), Subtract()) => leftMatrix.sub(rightMatrix)
      case (Ref("right", _), Ref("left", _), Subtract()) => rightMatrix.sub(leftMatrix)
      case (Ref("left", _), Ref("right", _), FloatingPointDivide()) => leftMatrix.div(rightMatrix)
      case (Ref("right", _), Ref("left", _), FloatingPointDivide()) => rightMatrix.div(leftMatrix)
      case _ => fatal(s"Binary operation not supported on two blockmatrices: ${Pretty(applyBinOp)}")
    }
  }
}

case class BlockMatrixUnaryOp(bm: BlockMatrixIR, applyUnaryOp: ApplyUnaryPrimOp) extends BlockMatrixIR {
  override def typ: BlockMatrixType = ???

  override def children: IndexedSeq[BaseIR] = ???

  override def copy(newChildren: IndexedSeq[BaseIR]): BaseIR = ???
}
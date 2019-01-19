package is.hail.expr.ir

import is.hail.HailContext
import is.hail.expr.types.BlockMatrixType
import is.hail.expr.types.virtual.TFloat64
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

  override def children: IndexedSeq[BaseIR] = Array(left, right, applyBinOp)

  override def copy(newChildren: IndexedSeq[BaseIR]): BaseIR = {
    assert(newChildren.length == 3)
    BlockMatrixElementWiseBinaryOp(
      newChildren(0).asInstanceOf[BlockMatrixIR],
      newChildren(1).asInstanceOf[BlockMatrixIR],
      newChildren(2).asInstanceOf[ApplyBinaryPrimOp])
  }

  override protected[ir] def execute(hc: HailContext): BlockMatrix = {
    val leftValue = left.execute(hc)
    val rightValue = right.execute(hc)

    (applyBinOp.l, applyBinOp.r, applyBinOp.op) match {
      case (Ref(_, _), Ref(_, _), Add()) => leftValue.add(rightValue)
      case (Ref(_, _), Ref(_, _), Multiply()) => leftValue.mul(rightValue)
      case (Ref("left", _), Ref("right", _), Subtract()) => leftValue.sub(rightValue)
      case (Ref("right", _), Ref("left", _), Subtract()) => rightValue.sub(leftValue)
      case (Ref("left", _), Ref("right", _), FloatingPointDivide()) => leftValue.div(rightValue)
      case (Ref("right", _), Ref("left", _), FloatingPointDivide()) => rightValue.div(leftValue)
      case _ => fatal(s"Binary operation not supported on two blockmatrices: ${Pretty(applyBinOp)}")
    }
  }
}

case class BlockMatrixAndValueElementWiseBinaryOp(
  child: BlockMatrixIR,
  applyBinOp: ApplyBinaryPrimOp) extends BlockMatrixIR {

  override def typ: BlockMatrixType = {
    (applyBinOp.l, applyBinOp.r) match {
      // No reshaping when broadcasting a scalar
      case (Ref(_, _), F64(_)) | (F64(_), Ref(_, _)) => child.typ
      // Add cases for local tensor when type exists
      case _ => fatal(s"Incompatible type for broadcasting operation: ${Pretty(applyBinOp)}")
    }
  }

  override def children: IndexedSeq[BaseIR] = Array(child, applyBinOp)

  override def copy(newChildren: IndexedSeq[BaseIR]): BaseIR = {
    assert(newChildren.length == 2)
    BlockMatrixAndValueElementWiseBinaryOp(
      newChildren(0).asInstanceOf[BlockMatrixIR],
      newChildren(1).asInstanceOf[ApplyBinaryPrimOp])
  }

  override protected[ir] def execute(hc: HailContext): BlockMatrix = {
    val bmValue = child.execute(hc)

    (applyBinOp.l, applyBinOp.r, applyBinOp.op) match {
      case (Ref(_, _), F64(x), Add()) => bmValue.scalarAdd(x)
      case (F64(x), Ref(_, _), Add()) => bmValue.scalarAdd(x)
      case (Ref(_, _), F64(x), Multiply()) => bmValue.scalarMul(x)
      case (F64(x), Ref(_, _), Multiply()) => bmValue.scalarMul(x)
      case (Ref(_, _), F64(x), Subtract()) => bmValue.scalarSub(x)
      case (F64(x), Ref(_, _), Subtract()) => bmValue.reverseScalarSub(x)
      case (Ref(_, _), F64(x), FloatingPointDivide()) => bmValue.scalarDiv(x)
      case (F64(x), Ref(_, _), FloatingPointDivide()) => bmValue.reverseScalarDiv(x)
    }
  }
}
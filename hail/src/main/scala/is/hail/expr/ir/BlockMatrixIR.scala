package is.hail.expr.ir

import is.hail.HailContext
import is.hail.expr.types.BlockMatrixType
import is.hail.expr.types.virtual.TFloat64
import is.hail.linalg.BlockMatrix
import is.hail.utils.fatal

object BlockMatrixIR {
  def resizeAfterBroadcast(leftDims: IndexedSeq[Long], rightDims: IndexedSeq[Long]): IndexedSeq[Long] = {
    val resultDims = Array[Long]()

    var i = 1
    while (i < Math.min(leftDims.length, rightDims.length)) {
      val leftIdx = leftDims.length - i - 1
      val rightIdx = rightDims.length - i - 1

      val leftDimLength = leftDims(leftIdx)
      val rightDimLength = rightDims(rightIdx)
      assert(leftDimLength == rightDimLength || leftDimLength == 1 || rightDimLength == 1)

      resultDims :+ Math.max(leftDimLength, rightDimLength)
      i += 1
    }

    // When the smaller dimensional object is exhausted,
    // the remaining dimensions are taken from the other object
    if (i < leftDims.length) {
      resultDims ++ leftDims.slice(0, i)
    } else if (i < rightDims.length) {
      resultDims ++ rightDims.slice(0, i)
    }

    resultDims
  }
}

abstract sealed class BlockMatrixIR extends BaseIR {
  def typ: BlockMatrixType

  protected[ir] def execute(hc: HailContext): BlockMatrix =
    fatal("tried to execute unexecutable IR:\n" + Pretty(this))
}

case class BlockMatrixRead(path: String) extends BlockMatrixIR {
  override def typ: BlockMatrixType = {
    val metadata = BlockMatrix.readMetadata(HailContext.get, path)
    BlockMatrixType(
      TFloat64(),
      IndexedSeq(metadata.nRows, metadata.nCols),
      metadata.blockSize,
      IndexedSeq(true, true))
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
    BlockMatrixType(
      TFloat64(),
      IndexedSeq(value.nRows, value.nCols),
      value.blockSize,
      IndexedSeq(true, true))
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
  override def typ: BlockMatrixType = {
    BlockMatrixType(
      left.typ.elementType,
      BlockMatrixIR.resizeAfterBroadcast(left.typ.shape, right.typ.shape),
      left.typ.blockSize,
      left.typ.dimsPartitioned)
  }

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
      case (Ref(_, _: TFloat64), Ref(_, _: TFloat64), Add()) => leftValue.add(rightValue)
      case (Ref(_, _: TFloat64), Ref(_, _: TFloat64), Multiply()) => leftValue.mul(rightValue)
      case (Ref(_, _: TFloat64), Ref(_, _: TFloat64), Subtract()) => leftValue.sub(rightValue)
      case (Ref(_, _: TFloat64), Ref(_, _: TFloat64), FloatingPointDivide()) => leftValue.div(rightValue)
      case _ => fatal(s"Binary operation not supported on two blockmatrices: ${Pretty(applyBinOp)}")
    }
  }
}

case class BlockMatrixBroadcastValue(
  child: BlockMatrixIR,
  applyBinOp: ApplyBinaryPrimOp) extends BlockMatrixIR {

  override def typ: BlockMatrixType = {
    (applyBinOp.l, applyBinOp.r) match {
      // No reshaping when broadcasting a scalar
      case (Ref(_, _: TFloat64), F64(_)) | (F64(_), Ref(_, _: TFloat64)) => child.typ
      // Add cases for local tensor when type exists
      case _ => fatal(s"Unsupported type for broadcasting operation: ${Pretty(applyBinOp)}")
    }
  }

  override def children: IndexedSeq[BaseIR] = Array(child, applyBinOp)

  override def copy(newChildren: IndexedSeq[BaseIR]): BaseIR = {
    assert(newChildren.length == 2)
    BlockMatrixBroadcastValue(
      newChildren(0).asInstanceOf[BlockMatrixIR],
      newChildren(1).asInstanceOf[ApplyBinaryPrimOp])
  }

  override protected[ir] def execute(hc: HailContext): BlockMatrix = {
    val bmValue = child.execute(hc)

    (applyBinOp.l, applyBinOp.r, applyBinOp.op) match {
      case (Ref(_, _: TFloat64), F64(x), Add()) => bmValue.scalarAdd(x)
      case (F64(x), Ref(_, _: TFloat64), Add()) => bmValue.scalarAdd(x)
      case (Ref(_, _: TFloat64), F64(x), Multiply()) => bmValue.scalarMul(x)
      case (F64(x), Ref(_, _: TFloat64), Multiply()) => bmValue.scalarMul(x)
      case (Ref(_, _: TFloat64), F64(x), Subtract()) => bmValue.scalarSub(x)
      case (F64(x), Ref(_, _: TFloat64), Subtract()) => bmValue.reverseScalarSub(x)
      case (Ref(_, _: TFloat64), F64(x), FloatingPointDivide()) => bmValue.scalarDiv(x)
      case (F64(x), Ref(_, _: TFloat64), FloatingPointDivide()) => bmValue.reverseScalarDiv(x)
    }
  }
}
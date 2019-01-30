package is.hail.expr.ir

import is.hail.HailContext
import is.hail.expr.types.BlockMatrixType
import is.hail.expr.types.virtual.TFloat64
import is.hail.linalg.BlockMatrix
import is.hail.utils.fatal

import breeze.linalg.DenseMatrix

object BlockMatrixIR {

  def resizeAfterMap(childShape: IndexedSeq[Long], left: IR, right: IR): IndexedSeq[Long] = {
    (left, right) match {
      // No reshaping when broadcasting a scalar
      case (Ref(_, _), F64(_)) | (F64(_), Ref(_, _)) => childShape
      // Reshaping when broadcasting a row or column vector
      case (MakeStruct(fields), Ref(_, _: TFloat64)) =>
        val (_, shape) = fields.head
        BlockMatrixIR.resizeAfterBroadcast(childShape, shape.asInstanceOf[IndexedSeq[Long]])
      case (Ref(_, _: TFloat64), MakeStruct(fields)) =>
        val (_, shape) = fields.head
        BlockMatrixIR.resizeAfterBroadcast(childShape, shape.asInstanceOf[IndexedSeq[Long]])
      case _ => fatal(s"Unsupported types for broadcasting operation: ${Pretty(left)}, ${Pretty(right)}")
    }
  }

  def resizeAfterBroadcast(leftShape: IndexedSeq[Long], rightShape: IndexedSeq[Long]): IndexedSeq[Long] = {
    var resultShape = IndexedSeq[Long]()

    for (i <- 0 until Math.min(leftShape.length, rightShape.length)) {
      val leftDimLength = leftShape(leftShape.length - i - 1)
      val rightDimLength = rightShape(rightShape.length - i - 1)
      assert(leftDimLength == rightDimLength || leftDimLength == 1 || rightDimLength == 1)

      resultShape = Math.max(leftDimLength, rightDimLength) +: resultShape
    }

    // When the smaller dimensional object is exhausted,
    // the remaining dimensions are taken from the other object
    if (rightShape.length < leftShape.length) {
      resultShape = leftShape.slice(0, leftShape.length - rightShape.length) ++ resultShape
    } else if (leftShape.length < rightShape.length) {
      resultShape = rightShape.slice(0, rightShape.length - leftShape.length) ++ resultShape
    }

    resultShape
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
    // Hack to get around Tensor x Tensor broadcasting
    // not being implemented in BlockMatrix
    if (left.typ.shape != right.typ.shape) {
      return executeAsBroadcastingValues(hc, left, right, applyBinOp.op)
    }

    (applyBinOp.l, applyBinOp.r, applyBinOp.op) match {
      case (Ref(_, _: TFloat64), Ref(_, _: TFloat64), Add()) =>
        left.execute(hc).add(right.execute(hc))
      case (Ref(_, _: TFloat64), Ref(_, _: TFloat64), Multiply()) =>
        left.execute(hc).mul(right.execute(hc))
      case (Ref(_, _: TFloat64), Ref(_, _: TFloat64), Subtract()) =>
        left.execute(hc).sub(right.execute(hc))
      case (Ref(_, _: TFloat64), Ref(_, _: TFloat64), FloatingPointDivide()) =>
        left.execute(hc).div(right.execute(hc))
      case _ => fatal(s"Binary operation not supported on two blockmatrices: ${Pretty(applyBinOp)}")
    }
  }

  def executeAsBroadcastingValues(
    hc: HailContext,
    left: BlockMatrixIR,
    right: BlockMatrixIR,
    op: BinaryOp): BlockMatrix = {

    val leftValue = left.execute(hc)
    val rightValue = right.execute(hc)

    (left.typ.shape, right.typ.shape) match {
      // Right scalar
      case (IndexedSeq(_, _), IndexedSeq(1, 1)) =>
        val rightAsScalar = rightValue.toBreezeMatrix().apply(0, 0)
        op match {
          case Add() => leftValue.scalarAdd(rightAsScalar)
          case Multiply() => leftValue.scalarMul(rightAsScalar)
          case Subtract() => leftValue.scalarSub(rightAsScalar)
          case FloatingPointDivide() => leftValue.scalarDiv(rightAsScalar)
        }
      // Left scalar
      case (IndexedSeq(1, 1), IndexedSeq(_, _)) =>
        val leftAsScalar = leftValue.toBreezeMatrix().apply(0, 0)
        op match {
          case Add() => rightValue.scalarAdd(leftAsScalar)
          case Multiply() => rightValue.scalarMul(leftAsScalar)
          case Subtract() => rightValue.reverseScalarSub(leftAsScalar)
          case FloatingPointDivide() => rightValue.reverseScalarDiv(leftAsScalar)
        }
      //Right row vector
      case (IndexedSeq(_, _), IndexedSeq(1, _)) =>
        val rightAsRowVec = rightValue.toBreezeMatrix().data
        op match {
          case Add() => leftValue.rowVectorAdd(rightAsRowVec)
          case Multiply() => leftValue.rowVectorMul(rightAsRowVec)
          case Subtract() => leftValue.rowVectorSub(rightAsRowVec)
          case FloatingPointDivide() => leftValue.rowVectorDiv(rightAsRowVec)
        }
      //Right col vector
      case (IndexedSeq(_, _), IndexedSeq(_, 1)) =>
        val rightAsColVec = rightValue.toBreezeMatrix().data
        op match {
          case Add() => leftValue.colVectorAdd(rightAsColVec)
          case Multiply() => leftValue.colVectorMul(rightAsColVec)
          case Subtract() => leftValue.colVectorSub(rightAsColVec)
          case FloatingPointDivide() => leftValue.colVectorDiv(rightAsColVec)
        }
      //Left row vector
      case (IndexedSeq(1, _), IndexedSeq(_, _)) =>
        val leftAsRowVec = leftValue.toBreezeMatrix().data
        op match {
          case Add() => rightValue.rowVectorAdd(leftAsRowVec)
          case Multiply() => rightValue.rowVectorMul(leftAsRowVec)
          case Subtract() => rightValue.reverseRowVectorSub(leftAsRowVec)
          case FloatingPointDivide() => rightValue.reverseRowVectorDiv(leftAsRowVec)
        }
      //Left col vector
      case (IndexedSeq(_, 1), IndexedSeq(_, _)) =>
        val leftAsColVec = leftValue.toBreezeMatrix().data
        op match {
          case Add() => rightValue.colVectorAdd(leftAsColVec)
          case Multiply() => rightValue.colVectorMul(leftAsColVec)
          case Subtract() => rightValue.reverseColVectorSub(leftAsColVec)
          case FloatingPointDivide() => rightValue.reverseColVectorDiv(leftAsColVec)
        }
    }
  }
}

case class BlockMatrixMap(
  child: BlockMatrixIR,
  applyBinOp: ApplyBinaryPrimOp) extends BlockMatrixIR {

  override def typ: BlockMatrixType = {
    BlockMatrixType(child.typ.elementType,
      BlockMatrixIR.resizeAfterMap(child.typ.shape, applyBinOp.l, applyBinOp.r),
      child.typ.blockSize,
      child.typ.dimsPartitioned)
  }

  override def children: IndexedSeq[BaseIR] = Array(child, applyBinOp)

  override def copy(newChildren: IndexedSeq[BaseIR]): BaseIR = {
    assert(newChildren.length == 2)
    BlockMatrixMap(
      newChildren(0).asInstanceOf[BlockMatrixIR],
      newChildren(1).asInstanceOf[ApplyBinaryPrimOp])
  }

  override protected[ir] def execute(hc: HailContext): BlockMatrix = {
    val bmValue = child.execute(hc)

    (applyBinOp.l, applyBinOp.r, applyBinOp.op) match {
      //Broadcasting by scalars
      case (Ref(_, _: TFloat64), F64(x), Add()) => bmValue.scalarAdd(x)
      case (F64(x), Ref(_, _: TFloat64), Add()) => bmValue.scalarAdd(x)
      case (Ref(_, _: TFloat64), F64(x), Multiply()) => bmValue.scalarMul(x)
      case (F64(x), Ref(_, _: TFloat64), Multiply()) => bmValue.scalarMul(x)
      case (Ref(_, _: TFloat64), F64(x), Subtract()) => bmValue.scalarSub(x)
      case (F64(x), Ref(_, _: TFloat64), Subtract()) => bmValue.reverseScalarSub(x)
      case (Ref(_, _: TFloat64), F64(x), FloatingPointDivide()) => bmValue.scalarDiv(x)
      case (F64(x), Ref(_, _: TFloat64), FloatingPointDivide()) => bmValue.reverseScalarDiv(x)
      //Broadcasting by vectors on the right
      case (Ref(_, _: TFloat64), MakeStruct(fields), op@_) =>
        val newRight = convertToLiteral(hc, fields)
        BlockMatrixElementWiseBinaryOp(child, newRight, ApplyBinaryPrimOp(op, applyBinOp.l, Ref("element", TFloat64()))).execute(hc)
      //Broadcasting by vectors on the left
      case (MakeStruct(fields), Ref(_, _: TFloat64), op@_) =>
        val newLeft = convertToLiteral(hc, fields)
        BlockMatrixElementWiseBinaryOp(newLeft, child, ApplyBinaryPrimOp(op, Ref("element", TFloat64()), applyBinOp.r)).execute(hc)
    }
  }

  private def convertToLiteral(hc: HailContext, vectorInfo: Seq[(String, IR)]): BlockMatrixLiteral = {
    val (_, shapeLiteral) = vectorInfo.head
    val (_, dataLiteral) = vectorInfo(1)
    val shape = shapeLiteral.asInstanceOf[Literal].value.asInstanceOf[Seq[Long]]
    val vector = dataLiteral.asInstanceOf[Literal].value.asInstanceOf[Seq[Double]].toArray
    val nRows = shape.head.toInt
    val nCols = shape(1).toInt

    val bm = BlockMatrix.fromBreezeMatrix(hc.sc,
      new DenseMatrix[Double](nRows, nCols, vector, 0, nCols, isTranspose = true), child.typ.blockSize)

    new BlockMatrixLiteral(bm)
  }
}
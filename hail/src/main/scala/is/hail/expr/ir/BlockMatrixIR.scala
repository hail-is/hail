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
        BlockMatrixIR.shapeAfterBroadcast(childShape, shape.asInstanceOf[IndexedSeq[Long]])
      case (Ref(_, _: TFloat64), MakeStruct(fields)) =>
        val (_, shape) = fields.head
        BlockMatrixIR.shapeAfterBroadcast(childShape, shape.asInstanceOf[IndexedSeq[Long]])
      case _ => fatal(s"Unsupported types for broadcasting operation: ${Pretty(left)}, ${Pretty(right)}")
    }
  }

  def shapeAfterBroadcast(leftShape: IndexedSeq[Long], rightShape: IndexedSeq[Long]): IndexedSeq[Long] = {
    val leftNDims = leftShape.length
    val rightNDims = rightShape.length

    var paddedLeft: IndexedSeq[Long] = null
    var paddedRight: IndexedSeq[Long] = null
    if (leftNDims < rightNDims) {
      paddedLeft = IndexedSeq.fill[Long](rightNDims - leftNDims)(1) ++ leftShape
      paddedRight = rightShape
    } else if (rightNDims < leftNDims) {
      paddedRight = IndexedSeq.fill[Long](leftNDims - rightNDims)(1) ++ rightShape
      paddedLeft = leftShape
    }

    (paddedLeft, paddedRight).zipped.map {
      (lDimLength, rDimLength) =>
        assert(lDimLength == rDimLength || lDimLength == 1 || rDimLength == 1)
        Math.max(lDimLength, rDimLength)
    }
  }

  def executeAsBroadcastingValues(left: BlockMatrix, right: BlockMatrix, op: BinaryOp): BlockMatrix = {
    val leftShape = IndexedSeq(left.nRows, left.nCols)
    val rightShape = IndexedSeq(right.nRows, right.nCols)

    (leftShape, rightShape) match {
      // Right scalar
      case (IndexedSeq(_, _), IndexedSeq(1, 1)) =>
        val rightAsScalar = right.toBreezeMatrix().apply(0, 0)
        op match {
          case Add() => left.scalarAdd(rightAsScalar)
          case Multiply() => left.scalarMul(rightAsScalar)
          case Subtract() => left.scalarSub(rightAsScalar)
          case FloatingPointDivide() => left.scalarDiv(rightAsScalar)
        }
      // Left scalar
      case (IndexedSeq(1, 1), IndexedSeq(_, _)) =>
        val leftAsScalar = left.toBreezeMatrix().apply(0, 0)
        op match {
          case Add() => right.scalarAdd(leftAsScalar)
          case Multiply() => right.scalarMul(leftAsScalar)
          case Subtract() => right.reverseScalarSub(leftAsScalar)
          case FloatingPointDivide() => right.reverseScalarDiv(leftAsScalar)
        }
      //Right row vector
      case (IndexedSeq(_, _), IndexedSeq(1, _)) =>
        val rightAsRowVec = right.toBreezeMatrix().data
        op match {
          case Add() => left.rowVectorAdd(rightAsRowVec)
          case Multiply() => left.rowVectorMul(rightAsRowVec)
          case Subtract() => left.rowVectorSub(rightAsRowVec)
          case FloatingPointDivide() => left.rowVectorDiv(rightAsRowVec)
        }
      //Right col vector
      case (IndexedSeq(_, _), IndexedSeq(_, 1)) =>
        val rightAsColVec = right.toBreezeMatrix().data
        op match {
          case Add() => left.colVectorAdd(rightAsColVec)
          case Multiply() => left.colVectorMul(rightAsColVec)
          case Subtract() => left.colVectorSub(rightAsColVec)
          case FloatingPointDivide() => left.colVectorDiv(rightAsColVec)
        }
      //Left row vector
      case (IndexedSeq(1, _), IndexedSeq(_, _)) =>
        val leftAsRowVec = left.toBreezeMatrix().data
        op match {
          case Add() => right.rowVectorAdd(leftAsRowVec)
          case Multiply() => right.rowVectorMul(leftAsRowVec)
          case Subtract() => right.reverseRowVectorSub(leftAsRowVec)
          case FloatingPointDivide() => right.reverseRowVectorDiv(leftAsRowVec)
        }
      //Left col vector
      case (IndexedSeq(_, 1), IndexedSeq(_, _)) =>
        val leftAsColVec = left.toBreezeMatrix().data
        op match {
          case Add() => right.colVectorAdd(leftAsColVec)
          case Multiply() => right.colVectorMul(leftAsColVec)
          case Subtract() => right.reverseColVectorSub(leftAsColVec)
          case FloatingPointDivide() => right.reverseColVectorDiv(leftAsColVec)
        }
    }
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
      BlockMatrixIR.shapeAfterBroadcast(left.typ.shape, right.typ.shape),
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
      val leftValue = left.execute(hc)
      val rightValue = right.execute(hc)
      return BlockMatrixIR.executeAsBroadcastingValues(leftValue, rightValue, applyBinOp.op)
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
    val childBM = child.execute(hc)

    (applyBinOp.l, applyBinOp.r, applyBinOp.op) match {
      //Broadcasting by scalars
      case (Ref(_, _: TFloat64), F64(x), Add()) => childBM.scalarAdd(x)
      case (F64(x), Ref(_, _: TFloat64), Add()) => childBM.scalarAdd(x)
      case (Ref(_, _: TFloat64), F64(x), Multiply()) => childBM.scalarMul(x)
      case (F64(x), Ref(_, _: TFloat64), Multiply()) => childBM.scalarMul(x)
      case (Ref(_, _: TFloat64), F64(x), Subtract()) => childBM.scalarSub(x)
      case (F64(x), Ref(_, _: TFloat64), Subtract()) => childBM.reverseScalarSub(x)
      case (Ref(_, _: TFloat64), F64(x), FloatingPointDivide()) => childBM.scalarDiv(x)
      case (F64(x), Ref(_, _: TFloat64), FloatingPointDivide()) => childBM.reverseScalarDiv(x)
      //Broadcasting vector on the right
      case (Ref(_, _: TFloat64), MakeStruct(fields), op@_) =>
        val rightBM = convertToBlockMatrix(hc, fields)
        BlockMatrixIR.executeAsBroadcastingValues(childBM, rightBM, op)
      //Broadcasting vector on the left
      case (MakeStruct(fields), Ref(_, _: TFloat64), op@_) =>
        val leftBM = convertToBlockMatrix(hc, fields)
        BlockMatrixIR.executeAsBroadcastingValues(leftBM, childBM, op)
    }
  }

  private def convertToBlockMatrix(hc: HailContext, vectorInfo: Seq[(String, IR)]): BlockMatrix = {
    val (_, shapeLiteral) = vectorInfo.head
    val (_, dataLiteral) = vectorInfo(1)
    val shape = shapeLiteral.asInstanceOf[Literal].value.asInstanceOf[Seq[Long]]
    val vector = dataLiteral.asInstanceOf[Literal].value.asInstanceOf[Seq[Double]].toArray
    val nRows = shape.head.toInt
    val nCols = shape(1).toInt

    BlockMatrix.fromBreezeMatrix(hc.sc,
      new DenseMatrix[Double](nRows, nCols, vector, 0, nCols, isTranspose = true), child.typ.blockSize)
  }
}
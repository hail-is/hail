package is.hail.expr.ir

import is.hail.HailContext
import is.hail.expr.types.BlockMatrixType
import is.hail.expr.types.virtual.TFloat64
import is.hail.linalg.BlockMatrix
import is.hail.utils.fatal
import breeze.linalg.DenseMatrix
import is.hail.expr.ir.Broadcast2D.Broadcast2D
import is.hail.expr.types.virtual.Type

object BlockMatrixIR {
  def toBlockMatrix(
    hc: HailContext,
    nRows: Int,
    nCols: Int,
    data: Array[Double],
    blockSize: Int = BlockMatrix.defaultBlockSize): BlockMatrix = {

    BlockMatrix.fromBreezeMatrix(hc.sc,
      new DenseMatrix[Double](nRows, nCols, data, 0, nCols, isTranspose = true), blockSize)
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

case class BlockMatrixMap2(
  left: BlockMatrixIR,
  right: BlockMatrixIR,
  applyBinOp: ApplyBinaryPrimOp) extends BlockMatrixIR {

  override def typ: BlockMatrixType = left.typ

  override def children: IndexedSeq[BaseIR] = Array(left, right, applyBinOp)

  override def copy(newChildren: IndexedSeq[BaseIR]): BaseIR = {
    assert(newChildren.length == 3)
    BlockMatrixMap2(
      newChildren(0).asInstanceOf[BlockMatrixIR],
      newChildren(1).asInstanceOf[BlockMatrixIR],
      newChildren(2).asInstanceOf[ApplyBinaryPrimOp])
  }

  override protected[ir] def execute(hc: HailContext): BlockMatrix = {
    val op = applyBinOp.op
    left match {
      case BlockMatrixBroadcast(scalarIR: BlockMatrixIR, Broadcast2D.SCALAR, _, _, _) =>
        scalarOnLeft(hc, coerceToScalar(hc, scalarIR), right, op)
      case BlockMatrixBroadcast(rowVectorIR: BlockMatrixIR, Broadcast2D.ROW, _, _, _) =>
        rowVectorOnLeft(hc, coerceToVector(hc, rowVectorIR), right, op)
      case BlockMatrixBroadcast(colVectorIR: BlockMatrixIR, Broadcast2D.COL, _, _, _) =>
        colVectorOnLeft(hc, coerceToVector(hc, colVectorIR), right, op)
      case _ =>
        val leftAsMatrix = left.execute(hc)
        matrixOnLeft(hc, leftAsMatrix, right, op)
    }
  }

  private def scalarOnLeft(hc: HailContext, scalar: Double, right: BlockMatrixIR, op: BinaryOp): BlockMatrix = {
    right match {
      case BlockMatrixBroadcast(vectorIR: BlockMatrixIR, Broadcast2D.ROW, _, _, _) =>
        val rightRowVector = coerceToVector(hc, vectorIR)
        val rowVectorAsBm = BlockMatrixIR.toBlockMatrix(hc, 1, rightRowVector.length, rightRowVector)
        opWithScalar(rowVectorAsBm, scalar, op, reverse = true)
      case BlockMatrixBroadcast(vectorIR: BlockMatrixIR, Broadcast2D.COL, _, _, _) =>
        val rightColVector = coerceToVector(hc, vectorIR)
        val colVectorAsBm = BlockMatrixIR.toBlockMatrix(hc, rightColVector.length, 1, rightColVector)
        opWithScalar(colVectorAsBm, scalar, op, reverse = true)
      case _ =>
        val rightValue = right.execute(hc)
        opWithScalar(rightValue, scalar, op, reverse = true)
    }
  }

  private def rowVectorOnLeft(hc: HailContext, rowVector: Array[Double], right: BlockMatrixIR, op: BinaryOp): BlockMatrix = {
    right match {
      case BlockMatrixBroadcast(scalarIR: BlockMatrixIR, Broadcast2D.SCALAR, _, _, _) =>
        val rightAsScalar = coerceToScalar(hc, scalarIR)
        val rowVectorAsBm = BlockMatrixIR.toBlockMatrix(hc, 1, rowVector.length, rowVector)
        opWithScalar(rowVectorAsBm, rightAsScalar, op, reverse = false)
      case _ =>
        val rightValue = right.execute(hc)
        opWithRowVector(rightValue, rowVector, op, reverse = true)
    }
  }

  private def colVectorOnLeft(hc: HailContext, colVector: Array[Double], right: BlockMatrixIR, op: BinaryOp): BlockMatrix = {
    right match {
      case BlockMatrixBroadcast(scalarIR: BlockMatrixIR, Broadcast2D.SCALAR, _, _, _) =>
        val rightAsScalar = coerceToScalar(hc, scalarIR)
        val colVectorAsBm = BlockMatrixIR.toBlockMatrix(hc, colVector.length, 1, colVector)
        opWithScalar(colVectorAsBm, rightAsScalar, op, reverse = false)
      case _ =>
        val rightValue = right.execute(hc)
        opWithColVector(rightValue, colVector, op, reverse = true)
    }
  }

  private def matrixOnLeft(hc: HailContext, matrix: BlockMatrix, right: BlockMatrixIR, op: BinaryOp): BlockMatrix = {
    right match {
      case BlockMatrixBroadcast(scalarIR: BlockMatrixIR, Broadcast2D.SCALAR, _, _, _) =>
        val rightAsScalar = coerceToScalar(hc, scalarIR)
        opWithScalar(matrix, rightAsScalar, op, reverse = false)
      case BlockMatrixBroadcast(vectorIR: BlockMatrixIR, Broadcast2D.ROW, _, _, _) =>
        val rightAsRowVec = coerceToVector(hc, vectorIR)
        opWithRowVector(matrix, rightAsRowVec, op, reverse = false)
      case BlockMatrixBroadcast(vectorIR: BlockMatrixIR, Broadcast2D.COL, _, _, _) =>
        val rightAsColVec = coerceToVector(hc, vectorIR)
        opWithColVector(matrix, rightAsColVec, op, reverse = false)
      case _ =>
        val rightValue = right.execute(hc)
        opWithTwoBlockMatrices(matrix, rightValue, op)
    }
  }

  private def coerceToScalar(hc: HailContext, ir: BlockMatrixIR): Double = {
    ir match {
      case ValueToBlockMatrix(child, _, _, _, _) =>
        Interpret[Any](child) match {
          case scalar: Double => scalar
          case oneElementArray: IndexedSeq[Double] => oneElementArray.head
        }
      case _ => ir.execute(hc).toBreezeMatrix().apply(0, 0)
    }
  }

  private def coerceToVector(hc: HailContext, ir: BlockMatrixIR): Array[Double] = {
    ir match {
      case ValueToBlockMatrix(child, _, _, _, _) =>
        Interpret[Any](child) match {
          case vector: IndexedSeq[Double] => vector.toArray
        }
      case _ => ir.execute(hc).toBreezeMatrix().data
    }
  }

  private def opWithScalar(left: BlockMatrix, right: Double, op: BinaryOp, reverse: Boolean): BlockMatrix = {
    op match {
      case Add() => left.scalarAdd(right)
      case Multiply() => left.scalarMul(right)
      case Subtract() =>
        if (reverse) left.reverseScalarSub(right) else left.scalarSub(right)
      case FloatingPointDivide() =>
        if (reverse) left.reverseScalarDiv(right) else left.scalarDiv(right)
    }
  }

  private def opWithRowVector(left: BlockMatrix, right: Array[Double], op: BinaryOp, reverse: Boolean): BlockMatrix = {
    op match {
      case Add() => left.rowVectorAdd(right)
      case Multiply() => left.rowVectorMul(right)
      case Subtract() =>
        if (reverse) left.reverseRowVectorSub(right) else left.rowVectorSub(right)
      case FloatingPointDivide() =>
        if (reverse) left.reverseRowVectorDiv(right) else left.rowVectorDiv(right)
    }
  }

  private def opWithColVector(left: BlockMatrix, right: Array[Double], op: BinaryOp, reverse: Boolean): BlockMatrix = {
    op match {
      case Add() => left.colVectorAdd(right)
      case Multiply() => left.colVectorMul(right)
      case Subtract() =>
        if (reverse) left.reverseColVectorSub(right) else left.colVectorSub(right)
      case FloatingPointDivide() =>
        if (reverse) left.reverseColVectorDiv(right) else left.colVectorDiv(right)
    }
  }

  private def opWithTwoBlockMatrices(left: BlockMatrix, right: BlockMatrix, op: BinaryOp): BlockMatrix = {
    op match {
      case Add() => left.add(right)
      case Multiply() => left.mul(right)
      case Subtract() => left.sub(right)
      case FloatingPointDivide() => left.div(right)
    }
  }
}

// This is now essentially only useful for unary operations
case class BlockMatrixMap(
  child: BlockMatrixIR,
  applyBinOp: ApplyBinaryPrimOp) extends BlockMatrixIR {

  override def typ: BlockMatrixType = child.typ

  override def children: IndexedSeq[BaseIR] = Array(child, applyBinOp)

  override def copy(newChildren: IndexedSeq[BaseIR]): BaseIR = {
    assert(newChildren.length == 2)
    BlockMatrixMap(
      newChildren(0).asInstanceOf[BlockMatrixIR],
      newChildren(1).asInstanceOf[ApplyBinaryPrimOp])
  }

  override protected[ir] def execute(hc: HailContext): BlockMatrix = {
    val childBM = child.execute(hc)

    //TODO Implement for Unary ops
    ???
  }
}

object Broadcast2D extends Enumeration {
  type Broadcast2D = Value
  val SCALAR, ROW, COL = Value

  def fromString(str: String): Broadcast2D = {
    str match {
      case "scalar" => SCALAR
      case "row" => ROW
      case "col" => COL
    }
  }
}

case class BlockMatrixBroadcast(
  child: BlockMatrixIR,
  broadcastType: Broadcast2D,
  shape: Array[Long],
  blockSize: Int,
  dimsPartitioned: Array[Boolean]) extends BlockMatrixIR {

  override def typ: BlockMatrixType = {
    BlockMatrixType(
      child.typ.elementType,
      shape,
      blockSize,
      dimsPartitioned
    )
  }

  override def children: IndexedSeq[BaseIR] = Array(child)

  override def copy(newChildren: IndexedSeq[BaseIR]): BaseIR = {
    assert(newChildren.length == 1)
    BlockMatrixBroadcast(newChildren(0).asInstanceOf[BlockMatrixIR], broadcastType, shape, blockSize, dimsPartitioned)
  }
}

case class ValueToBlockMatrix(
  child: IR,
  elementType: Type,
  shape: IndexedSeq[Long],
  blockSize: Int,
  dimsPartitioned: IndexedSeq[Boolean]) extends BlockMatrixIR {

  override def typ: BlockMatrixType = {
    BlockMatrixType(elementType, shape, blockSize, dimsPartitioned)
  }

  override def children: IndexedSeq[BaseIR] = Array(child)

  override def copy(newChildren: IndexedSeq[BaseIR]): BaseIR = {
    assert(newChildren.length == 1)
    ValueToBlockMatrix(newChildren(0).asInstanceOf[IR], elementType, shape, blockSize, dimsPartitioned)
  }

  override protected[ir] def execute(hc: HailContext): BlockMatrix = {
    Interpret[Any](child) match {
      case data: IndexedSeq[Double] =>
        BlockMatrixIR.toBlockMatrix(hc, shape(0).toInt, shape(1).toInt, data.toArray, blockSize)
    }
  }
}
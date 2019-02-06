package is.hail.expr.ir

import is.hail.HailContext
import is.hail.expr.types.BlockMatrixType
import is.hail.expr.types.virtual.{TArray, TFloat64}
import is.hail.linalg.BlockMatrix
import is.hail.utils.fatal
import breeze.linalg.DenseMatrix
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
    BlockMatrixType(TFloat64(), IndexedSeq(metadata.nRows, metadata.nCols), metadata.blockSize, IndexedSeq(true, true))
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
    BlockMatrixType(TFloat64(), IndexedSeq(value.nRows, value.nCols), value.blockSize, IndexedSeq(true, true))
  }

  override def children: IndexedSeq[BaseIR] = Array.empty[BlockMatrixIR]

  override def copy(newChildren: IndexedSeq[BaseIR]): BaseIR = {
    assert(newChildren.isEmpty)
    new BlockMatrixLiteral(value)
  }

  override protected[ir] def execute(hc: HailContext): BlockMatrix = value
}

case class BlockMatrixMap2(left: BlockMatrixIR, right: BlockMatrixIR, applyBinOp: ApplyBinaryPrimOp) extends BlockMatrixIR {

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
      case BlockMatrixBroadcast(scalarIR: BlockMatrixIR, IndexedSeq(), _, _, _, _) =>
        scalarOnLeft(hc, coerceToScalar(hc, scalarIR), right, op)
      case BlockMatrixBroadcast(vectorIR: BlockMatrixIR, IndexedSeq(i), outIndexExpr, _, _, _) =>
        val vector = coerceToVector(hc, vectorIR)
        outIndexExpr match {
          case IndexedSeq(_, `i`) => rowVectorOnLeft(hc, vector, right, op)
          case IndexedSeq(`i`, _) => colVectorOnLeft(hc, vector, right, op)
        }
      case _ =>
        matrixOnLeft(hc, left.execute(hc), right, op)
    }
  }

  private def scalarOnLeft(hc: HailContext, scalar: Double, right: BlockMatrixIR, op: BinaryOp): BlockMatrix = {
    right match {
      case BlockMatrixBroadcast(vectorIR: BlockMatrixIR, IndexedSeq(_), _, _, _, _) =>
        opWithScalar(vectorIR.execute(hc), scalar, op, reverse = true)
      case _ =>
        opWithScalar(right.execute(hc), scalar, op, reverse = true)
    }
  }

  private def rowVectorOnLeft(hc: HailContext, rowVector: Array[Double], right: BlockMatrixIR, op: BinaryOp): BlockMatrix = {
    right match {
      case BlockMatrixBroadcast(scalarIR: BlockMatrixIR, IndexedSeq(), _, _, _, _) =>
        val rightAsScalar = coerceToScalar(hc, scalarIR)
        val rowVectorAsBm = BlockMatrixIR.toBlockMatrix(hc, 1, rowVector.length, rowVector)
        opWithScalar(rowVectorAsBm, rightAsScalar, op, reverse = false)
      case _ =>
        opWithRowVector(right.execute(hc), rowVector, op, reverse = true)
    }
  }

  private def colVectorOnLeft(hc: HailContext, colVector: Array[Double], right: BlockMatrixIR, op: BinaryOp): BlockMatrix = {
    right match {
      case BlockMatrixBroadcast(scalarIR: BlockMatrixIR, IndexedSeq(), _, _, _, _) =>
        val rightAsScalar = coerceToScalar(hc, scalarIR)
        val colVectorAsBm = BlockMatrixIR.toBlockMatrix(hc, colVector.length, 1, colVector)
        opWithScalar(colVectorAsBm, rightAsScalar, op, reverse = false)
      case _ =>
        opWithColVector(right.execute(hc), colVector, op, reverse = true)
    }
  }

  private def matrixOnLeft(hc: HailContext, matrix: BlockMatrix, right: BlockMatrixIR, op: BinaryOp): BlockMatrix = {
    right match {
      case BlockMatrixBroadcast(scalarIR: BlockMatrixIR, IndexedSeq(), _, _, _, _) =>
        val rightAsScalar = coerceToScalar(hc, scalarIR)
        opWithScalar(matrix, rightAsScalar, op, reverse = false)
      case BlockMatrixBroadcast(vectorIR: BlockMatrixIR, IndexedSeq(i), outIndexExpr, _, _, _) =>
        outIndexExpr match {
          case IndexedSeq(_, `i`) =>
            val rightAsRowVec = coerceToVector(hc, vectorIR)
            opWithRowVector(matrix, rightAsRowVec, op, reverse = false)
          case IndexedSeq(`i`, _) =>
            val rightAsColVec = coerceToVector(hc, vectorIR)
            opWithColVector(matrix, rightAsColVec, op, reverse = false)
        }
      case _ =>
        opWithTwoBlockMatrices(matrix, right.execute(hc), op)
    }
  }

  private def coerceToScalar(hc: HailContext, ir: BlockMatrixIR): Double = {
    ir match {
      case ValueToBlockMatrix(child, _, _, _) =>
        Interpret[Any](child) match {
          case scalar: Double => scalar
          case oneElementArray: IndexedSeq[Double] => oneElementArray.head
        }
      case _ => ir.execute(hc).toBreezeMatrix().apply(0, 0)
    }
  }

  private def coerceToVector(hc: HailContext, ir: BlockMatrixIR): Array[Double] = {
    ir match {
      case ValueToBlockMatrix(child, _, _, _) =>
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

case class BlockMatrixBroadcast(
  child: BlockMatrixIR,
  inIndexExpr: IndexedSeq[String],
  outIndexExpr: IndexedSeq[String],
  shape: IndexedSeq[Long],
  blockSize: Int,
  dimsPartitioned: IndexedSeq[Boolean]) extends BlockMatrixIR {

  override def typ: BlockMatrixType = BlockMatrixType(child.typ.elementType, shape, blockSize, dimsPartitioned)

  override def children: IndexedSeq[BaseIR] = Array(child)

  override def copy(newChildren: IndexedSeq[BaseIR]): BaseIR = {
    assert(newChildren.length == 1)
    BlockMatrixBroadcast(newChildren(0).asInstanceOf[BlockMatrixIR], inIndexExpr, outIndexExpr,
      shape, blockSize, dimsPartitioned)
  }

  override protected[ir] def execute(hc: HailContext): BlockMatrix = {
    val childBm = child.execute(hc)
    val nRows = shape(0).toInt
    val nCols = shape(1).toInt

    inIndexExpr match {
      case IndexedSeq() =>
        val scalar = childBm.toBreezeMatrix().apply(0,0)
        BlockMatrix.fill(hc, nRows, nCols, scalar, blockSize)
      case IndexedSeq(i) =>
        outIndexExpr match {
          case IndexedSeq(_, `i`) =>
            val rowVector: Array[Double] = childBm.toBreezeMatrix().data
            var data = Array[Double]()
            (0 to nRows).foreach(_ => data = data ++ rowVector)
            BlockMatrixIR.toBlockMatrix(hc, nRows, nCols, data, blockSize)
          case IndexedSeq(`i`, _) =>
            val colVector = childBm.toBreezeMatrix().data
            var data = Array[Double]()
            (0 to nRows).foreach(idx => (0 to nCols).foreach(_ => data = data :+ colVector(idx)))
            BlockMatrixIR.toBlockMatrix(hc, nRows, nCols, data, blockSize)
        }
      case IndexedSeq(i, j) =>
        outIndexExpr match {
          case IndexedSeq(`j`, `i`) => childBm.transpose()
        }
    }
  }
}

case class ValueToBlockMatrix(
  child: IR,
  shape: IndexedSeq[Long],
  blockSize: Int,
  dimsPartitioned: IndexedSeq[Boolean]) extends BlockMatrixIR {

  override def typ: BlockMatrixType = BlockMatrixType(elementType(child.typ), shape, blockSize, dimsPartitioned)

  private def elementType(childType: Type): Type = {
    childType match {
      case array: TArray => array.elementType
      case _ => childType
    }
  }

  override def children: IndexedSeq[BaseIR] = Array(child)

  override def copy(newChildren: IndexedSeq[BaseIR]): BaseIR = {
    assert(newChildren.length == 1)
    ValueToBlockMatrix(newChildren(0).asInstanceOf[IR], shape, blockSize, dimsPartitioned)
  }

  override protected[ir] def execute(hc: HailContext): BlockMatrix = {
    assert(shape.length == 2)
    val nRows = shape(0).toInt
    val nCols = shape(1).toInt

    Interpret[Any](child) match {
      case scalar: Double =>
        BlockMatrix.fill(hc, nRows, nCols, scalar, blockSize)
      case data: IndexedSeq[Double] =>
        BlockMatrixIR.toBlockMatrix(hc, nRows, nCols, data.toArray, blockSize)
    }
  }
}
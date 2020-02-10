package is.hail.expr.types

import is.hail.utils._
import is.hail.expr.types.virtual.Type

object BlockMatrixType {
  def tensorToMatrixShape(shape: IndexedSeq[Long], isRowVector: Boolean): (Long, Long) = {
    shape match {
      case IndexedSeq() => (1, 1)
      case IndexedSeq(vectorLength) => if (isRowVector) (1, vectorLength) else (vectorLength, 1)
      case IndexedSeq(numRows, numCols) => (numRows, numCols)
    }
  }

  def matrixToTensorShape(nRows: Long,  nCols: Long): (IndexedSeq[Long], Boolean) = {
    (nRows, nCols) match {
      case (1, 1) => (FastIndexedSeq(), false)
      case (_, 1) => (FastIndexedSeq(nRows), false)
      case (1, _) => (FastIndexedSeq(nCols), true)
      case _ => (FastIndexedSeq(nRows, nCols), false)
    }
  }

  def numBlocks(n: Long, blockSize: Int): Int =
    java.lang.Math.floorDiv(n - 1, blockSize).toInt + 1

  def sparsityFromLinearBlocks(nCols: Long, nRows: Long, blockSize: Int, definedBlocks: Option[Array[Int]]): Array[Array[Boolean]] = {
    val nColBlocks = numBlocks(nCols, blockSize)
    val nRowBlocks = numBlocks(nRowBlocks, blockSize)

    definedBlocks.map { blocks =>
      val idxs = blocks.map { linearIdx => java.lang.Math.floorDiv(linearIdx, nColBlocks) -> linearIdx % nColBlocks }.toSet
      Array.tabulate(nRowBlocks)(i => Array.tabulate(nColBlocks)(j => idxs.contains(i -> j)))
    }.getOrElse(Array.fill(nRowBlocks)(Array.fill(nColBlocks)(true)))
  }

  // this is a shim method for the lowering.
  def apply(elementType: Type, shape: IndexedSeq[Long], isRowVector: Boolean, blockSize: Int): BlockMatrixType =
    BlockMatrixType(elementType, shape, isRowVector, blockSize, null)

  def dense(elementType: Type, nRows: Long, nCols: Long, blockSize: Int): BlockMatrixType = {
    val (shape, isRowVector) = matrixToTensorShape(nRows, nCols)
    val nRowBlocks = numBlocks(nRows, blockSize)
    val nColBlocks = numBlocks(nCols, blockSize)
    BlockMatrixType(elementType, shape, isRowVector, blockSize, Array.fill(nRowBlocks)(Array.fill(nColBlocks)(true)))
  }
}

case class BlockMatrixType(
  elementType: Type,
  shape: IndexedSeq[Long],
  isRowVector: Boolean,
  blockSize: Int,
  _definedBlocks: Array[Array[Boolean]]
) extends BaseType {

  lazy val (nRows: Long, nCols: Long) = BlockMatrixType.tensorToMatrixShape(shape, isRowVector)

  def matrixShape: (Long, Long) = nRows -> nCols

  lazy val nRowBlocks: Int = BlockMatrixType.numBlocks(nRows, blockSize)
  lazy val nColBlocks: Int = BlockMatrixType.numBlocks(nCols, blockSize)
  lazy val defaultBlockShape: (Int, Int) = (nRowBlocks, nColBlocks)

  lazy val definedBlocks: Array[Array[Boolean]] = {
    if (_definedBlocks == null)
      throw new UnsupportedOperationException("sparsity is not defined.")
    _definedBlocks
  }

  def getBlockIdx(i: Long): Int = java.lang.Math.floorDiv(i, blockSize).toInt

  val hasSparsity: Boolean = _definedBlocks != null
  lazy val isSparse: Boolean = _definedBlocks != null && definedBlocks.forall(rows => rows.forall(i => i))

  override def pretty(sb: StringBuilder, indent0: Int, compact: Boolean): Unit = {
    var indent = indent0

    val space: String = if (compact) "" else " "

    def newline() {
      if (!compact) {
        sb += '\n'
        sb.append(" " * indent)
      }
    }

    sb.append(s"BlockMatrix$space{")
    indent += 4
    newline()

    sb.append(s"elementType:$space")
    elementType.pretty(sb, indent, compact)
    sb += ','
    newline()

    sb.append(s"shape:$space[")
    shape.foreachBetween(dimSize => sb.append(dimSize))(sb.append(s",$space"))
    sb += ']'
    sb += ','
    newline()

    sb.append(s"isRowVector:$space")
    sb.append(isRowVector)
    sb += ','
    newline()

    sb.append(s"blockSize:$space")
    sb.append(blockSize)
    sb += ','
    newline()

    sb.append(s"definedBlocks:$space")
    if (hasSparsity && isSparse) {
      sb.append(definedBlocks.map(row => row.mkString("[", ",", "]")).mkString("[", ",", "]"))
    } else {
      sb.append("None")
    }
    sb += ','
    newline()

    indent -= 4
    newline()
    sb += '}'
  }
}

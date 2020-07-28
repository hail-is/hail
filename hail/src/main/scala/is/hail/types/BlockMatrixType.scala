package is.hail.types

import is.hail.utils._
import is.hail.types.virtual.{TFloat64, Type}
import is.hail.linalg.BlockMatrix

object BlockMatrixSparsity {
  private val builder: ArrayBuilder[(Int, Int)] = new ArrayBuilder[(Int, Int)]

  val dense: BlockMatrixSparsity = BlockMatrixSparsity(None)

  def apply(definedBlocks: IndexedSeq[(Int, Int)]): BlockMatrixSparsity = BlockMatrixSparsity(Some(definedBlocks))

  def apply(nRows: Int, nCols: Int)(exists: (Int, Int) => Boolean): BlockMatrixSparsity = {
    var i = 0
    builder.clear()
    while (i < nRows) {
      var j = 0
      while (j < nCols) {
        if (exists(i, j))
          builder += i -> j
        j += 1
      }
      i += 1
    }
    BlockMatrixSparsity(Some(builder.result().toFastIndexedSeq))
  }

  def fromLinearBlocks(nCols: Long, nRows: Long, blockSize: Int, definedBlocks: Option[IndexedSeq[Int]]): BlockMatrixSparsity = {
    val nColBlocks = BlockMatrixType.numBlocks(nCols, blockSize)
    definedBlocks.map { blocks =>
      BlockMatrixSparsity(blocks.map { linearIdx => java.lang.Math.floorDiv(linearIdx, nColBlocks) -> linearIdx % nColBlocks })
    }.getOrElse(dense)
  }
}

case class BlockMatrixSparsity(definedBlocks: Option[IndexedSeq[(Int, Int)]]) {
  lazy val definedBlocksColMajor: Option[IndexedSeq[(Int, Int)]] = definedBlocks.map { blocks =>
    blocks.sortWith { case ((i1, j1), (i2, j2)) =>
        j1 < j2 || (j1 == j2 && i1 < i2)
    }
  }

  def isSparse: Boolean = definedBlocks.isDefined
  lazy val blockSet: Set[(Int, Int)] = definedBlocks.get.toSet
  def hasBlock(idx: (Int, Int)): Boolean = definedBlocks.isEmpty || blockSet.contains(idx)
  def condense(blockOverlaps: => (Array[Array[Int]], Array[Array[Int]])): BlockMatrixSparsity = {
    definedBlocks.map { _ =>
      val (ro, co) = blockOverlaps
      BlockMatrixSparsity(ro.length, co.length) { (i, j) =>
        ro(i).exists(ii => co(j).exists(jj => hasBlock(ii -> jj)))
      }
    }.getOrElse(BlockMatrixSparsity.dense)
  }
  def allBlocks(nRowBlocks: Int, nColBlocks: Int, forceColMajor: Boolean): IndexedSeq[(Int, Int)] = {
    (if (forceColMajor) definedBlocksColMajor else definedBlocks).getOrElse {
      val foo = Array.fill[(Int, Int)](nRowBlocks * nColBlocks)(null)
      var i = 0
      while (i < nRowBlocks) {
        var j = 0
        while (j < nColBlocks) {
          foo(i + j * nRowBlocks) = i -> j
          j += 1
        }
        i += 1
      }
      foo
    }
  }
  override def toString: String =
    definedBlocks.map { blocks =>
      blocks.map { case (i, j) => s"($i,$j)" }.mkString("[", ",", "]")
    }.getOrElse("None")
}

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

  def getBlockIdx(i: Long, blockSize: Int): Int = java.lang.Math.floorDiv(i, blockSize).toInt

  def dense(elementType: Type, nRows: Long, nCols: Long, blockSize: Int): BlockMatrixType = {
    val (shape, isRowVector) = matrixToTensorShape(nRows, nCols)
    BlockMatrixType(elementType, shape, isRowVector, blockSize, BlockMatrixSparsity.dense)
  }

  def fromBlockMatrix(value: BlockMatrix): BlockMatrixType = {
    val sparsity = BlockMatrixSparsity.fromLinearBlocks(value.nRows, value.nCols, value.blockSize, value.gp.partitionIndexToBlockIndex)
    val (shape, isRowVector) = matrixToTensorShape(value.nRows, value.nCols)
    BlockMatrixType(TFloat64, shape, isRowVector, value.blockSize, sparsity)
  }
}

case class BlockMatrixType(
  elementType: Type,
  shape: IndexedSeq[Long],
  isRowVector: Boolean,
  blockSize: Int,
  sparsity: BlockMatrixSparsity
) extends BaseType {
  lazy val (nRows: Long, nCols: Long) = BlockMatrixType.tensorToMatrixShape(shape, isRowVector)

  def matrixShape: (Long, Long) = nRows -> nCols

  lazy val nRowBlocks: Int = BlockMatrixType.numBlocks(nRows, blockSize)
  lazy val nColBlocks: Int = BlockMatrixType.numBlocks(nCols, blockSize)
  lazy val defaultBlockShape: (Int, Int) = (nRowBlocks, nColBlocks)

  def getBlockIdx(i: Long): Int = java.lang.Math.floorDiv(i, blockSize).toInt
  def isSparse: Boolean = sparsity.isSparse
  def nDefinedBlocks: Int =
    if (isSparse) sparsity.definedBlocks.get.length else nRowBlocks * nColBlocks
  def hasBlock(idx: (Int, Int)): Boolean = {
    if (isSparse) sparsity.hasBlock(idx) else true
  }
  def allBlocks(forceColMajor: Boolean): IndexedSeq[(Int, Int)] = sparsity.allBlocks(nRowBlocks, nColBlocks, forceColMajor)

  lazy val linearizedDefinedBlocks: Option[IndexedSeq[Int]] = sparsity.definedBlocksColMajor.map { blocks =>
    blocks.map { case (i, j) => i + j * nRowBlocks }
  }

  def blockShape(i: Int, j: Int): (Long, Long) = {
    val r = if (i == nRowBlocks - 1) nRows - (i * blockSize) else blockSize
    val c = if (i == nColBlocks - 1) nCols - (i * blockSize) else blockSize
    r -> c
  }

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

    sb.append(s"sparsity:$space")
    sb.append(sparsity.toString)
    sb += ','
    newline()

    indent -= 4
    newline()
    sb += '}'
  }
}

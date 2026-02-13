package is.hail.types.virtual

import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.ir._
import is.hail.expr.ir.defs.{I64, If}
import is.hail.linalg.{BlockMatrix, MatrixSparsity}
import is.hail.utils.fatal

import scala.collection.compat._

import org.json4s.{JInt, JObject, JString, JValue}

object BlockMatrixType {
  def numBlocks(n: Long, blockSize: Int): Int =
    java.lang.Math.floorDiv(n - 1, blockSize).toInt + 1

  def getBlockIdx(i: Long, blockSize: Int): Int = java.lang.Math.floorDiv(i, blockSize).toInt

  def dense(elementType: Type, nRows: Long, nCols: Long, blockSize: Int): BlockMatrixType = {
    val nRowBlocks = numBlocks(nRows, blockSize)
    val nColBlocks = numBlocks(nCols, blockSize)
    val sparsity = MatrixSparsity.dense(nRowBlocks, nColBlocks)
    BlockMatrixType(elementType, nRows, nCols, blockSize, sparsity)
  }

  def fromBlockMatrix(value: BlockMatrix): BlockMatrixType = {
    val sparsity = MatrixSparsity.fromLinearCoords(
      numBlocks(value.nRows, value.blockSize),
      numBlocks(value.nCols, value.blockSize),
      value.gp.partitionIndexToBlockIndex,
    )
    BlockMatrixType(TFloat64, value.nRows, value.nCols, value.blockSize, sparsity)
  }

  def getBlockDependencies(keep: IterableOnce[IndexedSeq[Long]], blockSize: Int)
    : IndexedSeq[IndexedSeq[Int]] =
    keep.iterator.map(_.map(i => (i / blockSize).toInt).distinct).to(ArraySeq)
}

case class BlockMatrixType(
  elementType: Type,
  nRows: Long,
  nCols: Long,
  blockSize: Int,
  sparsity: MatrixSparsity,
) extends VType {
  require(blockSize >= 0)
  if (nRows == 0) fatal("block matrix must have at least one row")
  if (nCols == 0) fatal("block matrix must have at least one column")

  def nRowBlocks: Int = sparsity.nRows
  def nColBlocks: Int = sparsity.nCols

  def densify: BlockMatrixType = copy(sparsity = MatrixSparsity.Dense(nRowBlocks, nColBlocks))

  def getBlockIdx(i: Long): Int = java.lang.Math.floorDiv(i, blockSize).toInt
  def isSparse: Boolean = sparsity.isSparse

  def nDefinedBlocks: Int = sparsity.numDefined

  def hasBlock(row: Int, col: Int): Boolean = sparsity.contains(row, col)

  def transpose: BlockMatrixType = copy(nRows = nCols, nCols = nRows, sparsity = sparsity.transpose)

  def blockShape(i: Int, j: Int): (Long, Long) = {
    val r = if (i == nRowBlocks - 1) nRows - (i * blockSize) else blockSize.toLong
    val c = if (j == nColBlocks - 1) nCols - (j * blockSize) else blockSize.toLong
    r -> c
  }

  def blockShapeIR(i: IR, j: IR): (IR, IR) = {
    assert(i.typ == TInt32)
    assert(j.typ == TInt32)
    val r = If(i.ceq(nRowBlocks - 1), I64(nRows) - (i.toL * blockSize.toLong), blockSize.toLong)
    val c = If(j.ceq(nColBlocks - 1), I64(nCols) - (j.toL * blockSize.toLong), blockSize.toLong)
    r -> c
  }

  override def pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = {
    val space: String = if (compact) "" else " "
    val newline: String = if (compact) "" else "\n"
    val padding: String = if (compact) "" else " " * indent

    sb ++= "BlockMatrix" ++= space += '{' ++= newline: Unit

    sb ++= padding ++= "elementType:" ++= space: Unit
    elementType.pretty(sb, indent + 4, compact)
    sb += ',' ++= newline: Unit

    sb ++= padding ++= "shape:" ++= space += '[': Unit
    sb ++= nRows.toString += ',' ++= space ++= nCols.toString ++= "]," ++= newline: Unit

    sb ++= padding ++= "blockSize:" ++= space ++= s"$blockSize" += ',' ++= newline: Unit
    sb ++= padding ++= "sparsity:" ++= space ++= s"$sparsity" += ',' ++= newline: Unit

    sb += '}'
  }

  override def toJSON: JValue =
    JObject(
      "element_type" -> JString(elementType.toString),
      "n_rows" -> JInt(nRows),
      "n_cols" -> JInt(nCols),
      "block_size" -> JInt(blockSize),
    )
}

package is.hail.linalg

import org.apache.spark.Partitioner

case class GridPartitioner(blockSize: Int, nRows: Long, nCols: Long) extends Partitioner {
  require(nRows > 0 && nRows <= Int.MaxValue.toLong * blockSize)
  require(nCols > 0 && nCols <= Int.MaxValue.toLong * blockSize)
  
  def blockIndex(index: Long): Int = (index / blockSize).toInt

  def blockOffset(index: Long): Int = (index % blockSize).toInt

  val nBlockRows: Int = (nRows / blockSize).toInt
  val nBlockCols: Int = (nCols / blockSize).toInt
  
  val lastBlockRowNRows: Int = blockOffset(nRows - 1) + 1
  val lastBlockColNCols: Int = blockOffset(nCols - 1) + 1  
  
  def blockRowNRows(i: Int): Int = if (i < nBlockRows - 1) blockSize else lastBlockRowNRows
  def blockColNCols(j: Int): Int = if (j < nBlockCols - 1) blockSize else lastBlockColNCols

  def blockBlockRow(pi: Int): Int = pi % nBlockRows
  def blockBlockCol(pi: Int): Int = pi / nBlockRows

  def blockDims(pi: Int): (Int, Int) = (blockRowNRows(blockBlockRow(pi)), blockColNCols(blockBlockCol(pi)))
  
  def blockCoordinates(pi: Int): (Int, Int) = (blockBlockRow(pi), blockBlockCol(pi))

  def coordinatesBlock(i: Int, j: Int): Int = {
    require(0 <= i && i < nBlockRows, s"Block row $i out of range [0, $nBlockRows).")
    require(0 <= j && j < nBlockCols, s"Block column $j out of range [0, $nBlockCols).")
    i + j * nBlockRows
  }

  override val numPartitions: Int = nBlockRows * nBlockCols
  
  override def getPartition(key: Any): Int = key match {
    case (i: Int, j: Int) => coordinatesBlock(i, j)
  }
  
  def transpose: GridPartitioner = GridPartitioner(this.blockSize, this.nCols, this.nRows)
}

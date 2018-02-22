package is.hail.linalg

import org.apache.spark.Partitioner

case class GridPartitioner(blockSize: Int, nRows: Long, nCols: Long) extends Partitioner {
  require(nRows > 0)
  require(nCols > 0)
  
  def blockIndex(index: Long): Int = (index / blockSize).toInt
  def blockOffset(index: Long): Int = (index % blockSize).toInt  

  require((nRows - 1) / blockSize + 1 <= Int.MaxValue)
  val nBlockRows: Int = blockIndex(nRows - 1) + 1

  require((nCols - 1) / blockSize + 1 <= Int.MaxValue)    
  val nBlockCols: Int = blockIndex(nCols - 1) + 1

  require(nBlockRows.toLong * nBlockCols <= Int.MaxValue)
  override val numPartitions: Int = nBlockRows * nBlockCols
  
  def blockRowNRows(i: Int): Int =
    if (i < nBlockRows - 1)
      blockSize
    else
      blockOffset(nRows - 1) + 1

  def blockColNCols(j: Int): Int =
    if (j < nBlockCols - 1)
      blockSize
    else
      blockOffset(nRows - 1) + 1

  def blockDims(pi: Int): (Int, Int) = (blockRowNRows(blockBlockRow(pi)), blockColNCols(blockBlockCol(pi)))

  override def getPartition(key: Any): Int = key match {
    case (i: Int, j: Int) => coordinatesBlock(i, j)
  }

  def blockBlockRow(pi: Int): Int = pi % nBlockRows

  def blockBlockCol(pi: Int): Int = pi / nBlockRows

  def blockCoordinates(pi: Int): (Int, Int) = (blockBlockRow(pi), blockBlockCol(pi))

  def coordinatesBlock(i: Int, j: Int): Int = {
    require(0 <= i && i < nBlockRows, s"Block row $i out of range [0, $nBlockRows).")
    require(0 <= j && j < nBlockCols, s"Block column $j out of range [0, $nBlockCols).")
    i + j * nBlockRows
  }

  def transpose: GridPartitioner =
    GridPartitioner(this.blockSize, this.nCols, this.nRows)
}

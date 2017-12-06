package is.hail.distributedmatrix

import org.apache.spark.Partitioner

case class GridPartitioner(blockSize: Int, nRows: Long, nCols: Long) extends Partitioner {
  require(nRows > 0)
  require(nCols > 0)
  require((nRows - 1) / blockSize + 1 < Int.MaxValue)
  require((nCols - 1) / blockSize + 1 < Int.MaxValue)

  val nBlockRows: Int = ((nRows - 1) / blockSize + 1).toInt
  val nBlockCols: Int = ((nCols - 1) / blockSize + 1).toInt

  def blockRowNRows(i: Int): Int =
    if (i < nBlockRows - 1)
      blockSize
    else
      (nRows - (nBlockRows - 1) * blockSize).toInt

  def blockColNCols(j: Int): Int =
    if (j < nBlockCols - 1)
      blockSize
    else
      (nCols - (nBlockCols - 1) * blockSize).toInt

  def blockDims(pi: Int): (Int, Int) = (blockRowNRows(blockBlockRow(pi)), blockColNCols(blockBlockCol(pi)))

  override val numPartitions: Int = nBlockRows * nBlockCols
  assert(numPartitions >= nBlockRows && numPartitions >= nBlockCols)

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

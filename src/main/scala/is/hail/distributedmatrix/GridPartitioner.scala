package is.hail.distributedmatrix

import is.hail.utils.ArrayBuilder
import org.apache.spark.Partitioner

case class GridPartitioner(blockSize: Int, nRows: Long, nCols: Long) extends Partitioner {
  require(nRows > 0)
  require(nCols > 0)
  require((nRows + blockSize - 1) / blockSize < Int.MaxValue)
  require((nCols + blockSize - 1) / blockSize < Int.MaxValue)

  val nBlockRows: Int = blockIndex(nRows)
  val nBlockCols: Int = blockIndex(nCols)

  def blockIndex(index: Long): Int = ((index + blockSize - 1) / blockSize).toInt
  
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
  
  // 0, 0 gives diagonal blocks
  def bandedBlocks(lowerBandwidth: Long, upperBandwidth: Long): Array[Int] = {
    assert(lowerBandwidth >= 0 && upperBandwidth >= 0)
    
    val lowerBlockBandwidth = blockIndex(lowerBandwidth)
    val upperBlockBandwidth = blockIndex(upperBandwidth)

    val blocks = new ArrayBuilder[Int]
    
    var j = 0
    while (j < nBlockCols) {
      val offset = j * nBlockRows
      var i = (j - upperBlockBandwidth) max 0
      while (i <= ((j + lowerBlockBandwidth) min (nBlockRows - 1))) {
        blocks += offset + i
        i += 1
      }
      j += 1
    }
    
    blocks.result()
  }
  
  def lowerTriangularBlocks(): Array[Int] = {
    assert(nBlockRows <= nBlockCols)
    
    bandedBlocks(nRows, 0)
  }
  
  def transpose: GridPartitioner =
    GridPartitioner(this.blockSize, this.nCols, this.nRows)
}

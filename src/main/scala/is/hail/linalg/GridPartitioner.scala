package is.hail.linalg

import is.hail.HailContext
import is.hail.utils._
import org.apache.spark.Partitioner
import org.json4s.jackson

object GridPartitioner {
  // FIXME: hacky, uri is for BlockMatrix
  def read(hc: HailContext, uri: String): GridPartitioner = {
    val BlockMatrixMetadata(blockSize, nRows, nCols) =
      hc.hadoopConf.readTextFile(uri + BlockMatrix.metadataRelativePath) { isr =>
        implicit val formats = defaultJSONFormats
        jackson.Serialization.read[BlockMatrixMetadata](isr)
      }
    
    GridPartitioner(blockSize, nRows, nCols)
  }
}

case class GridPartitioner(blockSize: Int, nRows: Long, nCols: Long) extends Partitioner {
  require(nRows > 0 && nRows <= Int.MaxValue.toLong * blockSize)
  require(nCols > 0 && nCols <= Int.MaxValue.toLong * blockSize)
  
  def blockIndex(index: Long): Int = (index / blockSize).toInt

  def blockOffset(index: Long): Int = (index % blockSize).toInt

  val nBlockRows: Int = blockIndex(nRows - 1) + 1
  val nBlockCols: Int = blockIndex(nCols - 1) + 1
  
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
  
  // returns all blocks intersecting the diagonal band consisting of all entries
  //   with -lowerBandwidth <= (colIndex - rowIndex) <= upperBandwidth
  def bandedBlocks(lowerBandwidth: Long, upperBandwidth: Long): Array[Int] = {
    require(lowerBandwidth >= 0 && upperBandwidth >= 0)
    
    val lowerBlockBandwidth = blockIndex(lowerBandwidth + blockSize - 1)
    val upperBlockBandwidth = blockIndex(upperBandwidth + blockSize - 1)

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
  
  // returns all blocks intersecting the rectangle [firstRow, lastRow] x [firstCol, lastCol]
  def rectangularBlocks(firstRow: Long, lastRow: Long, firstCol: Long, lastCol: Long): Array[Int] = {
    require(firstRow >= 0 && firstRow <= lastRow && lastRow <= nRows)
    require(firstCol >= 0 && firstCol <= lastCol && lastCol <= nCols)
    
    val firstBlockRow = blockIndex(firstRow)
    val lastBlockRow = blockIndex(lastRow)
    val firstBlockCol = blockIndex(firstCol)
    val lastBlockCol = blockIndex(lastCol)

    val blocks = new Array[Int]((lastBlockRow - firstBlockRow + 1) * (lastBlockCol - firstBlockCol + 1))
    
    var k = 0
    var j = firstBlockCol
    while (j <= lastBlockCol) {
      val offset = j * nBlockRows
      var i = firstBlockRow
      while (i <= lastBlockRow) {
        blocks(k) = offset + i
        k += 1
        i += 1
      }
      j += 1
    }
    
    blocks
  }
  
  // returns all blocks intersecting the union of rectangles
  def rectangularBlocks(rectangles: Array[Array[Long]]): Array[Int] = {
    val keep = new Array[Boolean](numPartitions)
    
    rectangles.foreach { r =>
      assert(r.length == 4)
      val rBlocks = rectangularBlocks(r(0), r(1), r(2), r(3))
      var i = 0
      while (i < rBlocks.length) {
        keep(rBlocks(i)) = true
        i += 1
      }
    }
    
    val blocks = new ArrayBuilder[Int]()
    var block = 0
    while (block < numPartitions) {
      if (keep(block))
        blocks += block
      block += 1
    }
    
    blocks.result()
  }
}

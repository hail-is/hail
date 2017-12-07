package is.hail.distributedmatrix

import org.apache.spark.Partitioner

case class GridPartitioner(blockSize: Int, rows: Long, cols: Long) extends Partitioner {
  require(rows > 0)
  require(cols > 0)
  require((rows - 1) / blockSize + 1 < Int.MaxValue)
  require((cols - 1) / blockSize + 1 < Int.MaxValue)

  val rowPartitions: Int = ((rows - 1) / blockSize + 1).toInt
  val colPartitions: Int = ((cols - 1) / blockSize + 1).toInt
  
  val truncatedBlockRow: Int = (rows / blockSize).toInt
  val truncatedBlockCol: Int = (cols / blockSize).toInt
  
  val excessRows: Int = (rows % blockSize).toInt
  val excessCols: Int = (cols % blockSize).toInt
  
  def rowPartitionRows(i: Int): Int = if (i != truncatedBlockRow) blockSize else excessRows
  def colPartitionCols(j: Int): Int = if (j != truncatedBlockCol) blockSize else excessCols
  
  override val numPartitions: Int = rowPartitions * colPartitions
  assert(numPartitions >= rowPartitions && numPartitions >= colPartitions)
  
  override def getPartition(key: Any): Int = key match {
    case (i: Int, j: Int) => partitionIdFromBlockIndices(i, j)
  }

  def blockRowIndex(pid: Int): Int = pid % rowPartitions
  def blockColIndex(pid: Int): Int = pid / rowPartitions
  def blockCoordinates(pid: Int): (Int, Int) = (blockRowIndex(pid), blockColIndex(pid))

  def partitionIdFromBlockIndices(i: Int, j: Int): Int = {
    require(0 <= i && i < rowPartitions, s"Block row index $i out of range [0, $rowPartitions).")
    require(0 <= j && j < colPartitions, s"Block column index $j out of range [0, $colPartitions).")
    i + j * rowPartitions
  }

  def partitionIdFromGlobalIndices(i: Int, j: Int): Int = {
    require(0 <= i && i < rows, s"Row index $i out of range [0, $rows).")
    require(0 <= j && j < cols, s"Column index $j out of range [0, $cols).")
    partitionIdFromBlockIndices(i / blockSize, j / blockSize)
  }
  
  def transpose: GridPartitioner =
    GridPartitioner(this.blockSize, this.cols, this.rows)
}

package is.hail.distributedmatrix

import org.apache.spark.Partitioner

case class HailGridPartitioner(rows: Long, cols: Long, blockSize: Int, transposed: Boolean = false) extends Partitioner {
  require(rows > 0)
  require(cols > 0)
  require((rows - 1) / blockSize + 1 < Int.MaxValue)
  require((cols - 1) / blockSize + 1 < Int.MaxValue)

  val rowPartitions: Int = ((rows - 1) / blockSize + 1).toInt
  val colPartitions: Int = ((cols - 1) / blockSize + 1).toInt

  override val numPartitions: Int = rowPartitions * colPartitions

  override def getPartition(key: Any): Int = key match {
    case (i: Int, j: Int) => partitionIdFromBlockIndices(i, j)
  }

  def blockRowIndex(pid: Int) =
    if (transposed) pid / colPartitions else pid % rowPartitions
  def blockColIndex(pid: Int) =
    if (transposed) pid % colPartitions else pid / rowPartitions
  def blockCoordinates(pid: Int) =
    (blockRowIndex(pid), blockColIndex(pid))

  def partitionIdFromBlockIndices(i: Int, j: Int): Int = {
    require(0 <= i && i < rowPartitions, s"Row index $i out of range [0, $rowPartitions).")
    require(0 <= j && j < colPartitions, s"Column index $j out of range [0, $colPartitions).")
    if (transposed) {
      j + i * colPartitions
    } else {
      i + j * rowPartitions
    }
  }

  def partitionIdFromGlobalIndices(i: Int, j: Int): Int = {
    require(0 <= i && i < rows, s"Row index $i out of range [0, $rows).")
    require(0 <= j && j < cols, s"Column index $j out of range [0, $cols).")
    partitionIdFromBlockIndices(i / blockSize, j / blockSize)
  }

  def transpose: HailGridPartitioner =
    HailGridPartitioner(this.cols, this.rows, this.blockSize, !this.transposed)
}

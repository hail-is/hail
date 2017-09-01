package is.hail.distributedmatrix

import org.apache.spark.Partitioner

case class HailGridPartitioner(rows: Long, cols: Long, blockSize: Int, transposed: Boolean = false) extends Partitioner {
  require(rows > 0)
  require(cols > 0)
  require((rows - 1) / blockSize + 1 < Int.MaxValue)
  require((cols - 1) / blockSize + 1 < Int.MaxValue)

  private val untransposedRowPartitions = ((rows - 1) / blockSize + 1).toInt
  private val untransposedColPartitions = ((cols - 1) / blockSize + 1).toInt

  val rowPartitions = if (transposed) untransposedColPartitions else untransposedRowPartitions
  val colPartitions = if (transposed) untransposedRowPartitions else untransposedColPartitions

  override val numPartitions: Int = rowPartitions * colPartitions

  override def getPartition(key: Any): Int = key match {
    case (i: Int, j: Int) => partitionIdFromBlockIndices(i, j)
  }

  def blockRowIndex(pid: Int) =
    if (transposed) pid / rowPartitions else pid % rowPartitions
  def blockColIndex(pid: Int) =
    if (transposed) pid % rowPartitions else pid / rowPartitions
  def blockCoordinates(pid: Int) =
    (blockRowIndex(pid), blockColIndex(pid))

  def partitionIdFromBlockIndices(i: Int, j: Int): Int = {
    if (transposed) {
      require(0 <= j && j < untransposedRowPartitions, s"Row index $j out of range [0, $untransposedRowPartitions).")
      require(0 <= i && i < untransposedColPartitions, s"Column index $i out of range [0, $untransposedColPartitions).")
      j + i * untransposedRowPartitions
    } else {
      require(0 <= i && i < untransposedRowPartitions, s"Row index $i out of range [0, $untransposedRowPartitions).")
      require(0 <= j && j < untransposedColPartitions, s"Column index $j out of range [0, $untransposedColPartitions).")
      i + j * untransposedRowPartitions
    }
  }

  def partitionIdFromGlobalIndices(i: Int, j: Int): Int = {
    if (transposed) {
      require(0 <= j && j < rows, s"Row index $j out of range [0, $rows).")
      require(0 <= i && i < cols, s"Column index $i out of range [0, $cols).")
      partitionIdFromBlockIndices(j / blockSize, i / blockSize)
    } else {
      require(0 <= i && i < rows, s"Row index $i out of range [0, $rows).")
      require(0 <= j && j < cols, s"Column index $j out of range [0, $cols).")
      partitionIdFromBlockIndices(i / blockSize, j / blockSize)
    }
  }
}

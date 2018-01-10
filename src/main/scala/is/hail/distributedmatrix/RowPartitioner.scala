package is.hail.distributedmatrix

import is.hail.utils.BinarySearch
import org.apache.spark.Partitioner

case class RowPartitioner(partitionStarts: Array[Long]) extends Partitioner {
  override val numPartitions: Int = partitionStarts.length - 1

  override def getPartition(key: Any): Int = key match {
    case i: Long => BinarySearch.binarySearchInterval(partitionStarts, i)
  }
}

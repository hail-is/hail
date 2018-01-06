package is.hail.distributedmatrix

import java.util

import org.apache.spark.Partitioner

object RowPartitioner {
  def rangeBoundsFromStarts(partitionStarts: Array[Long]): Array[Long] = {
    val rangeBounds = new Array[Long](partitionStarts.length - 2)
    var pi = 0
    while (pi < partitionStarts.length - 2) {
      rangeBounds(pi) = partitionStarts(pi + 1) - 1
      pi += 1
    }
    rangeBounds
  }

  def fromPartitionStarts(partitionStarts: Array[Long]): RowPartitioner = {
    RowPartitioner(rangeBoundsFromStarts(partitionStarts))
  }
}

case class RowPartitioner(rangeBounds: Array[Long]) extends Partitioner {
  override val numPartitions: Int = rangeBounds.length + 1

  override def getPartition(key: Any): Int = key match {
    case i: Long =>
      var pi = 0
      if (rangeBounds.length < 128) {
        while (pi < rangeBounds.length && i > rangeBounds(pi)) {
          pi += 1
        }
      } else {
        pi = util.Arrays.binarySearch(rangeBounds, i)
        if (pi < 0) {
          // found no match, returned -[insertion point] - 1
          pi = -(pi + 1)
        } else if (pi < rangeBounds.length) {
          // found a match, decrement to first match to precede empty partitions
          while (pi > 0 && rangeBounds(pi) == rangeBounds(pi - 1))
            pi -= 1
        }
      }
      pi
  }
}

package is.hail.linalg

import org.apache.spark.Partitioner

object RowPartitioner {

  /** a represents a partitioning of the (mathematical) integers into intervals with interval j
    * given by [a(j), a(j+1)).
    *
    * -infty -1 a(0) 0 a(1) 1 len-2 a(len-1) len-1 infty (-----------)[---------)[-------- ...
    * ------------)[-----------------)
    *
    * a must be non-decreasing; repeated values correspond to empty intervals.
    *
    * Returns interval containing key: -1 iff a is empty or key < a(0) j iff a(j) <= key < a(j + 1)
    * len-1 iff a(len - 1) < key
    */
  def findInterval(a: Array[Long], key: Long): Int = {
    var lo = 0
    var hi = a.length - 1
    while (lo <= hi) {
      val mid = (lo + hi) >>> 1
      if (key < a(mid))
        hi = mid - 1
      else
        lo = mid + 1
    }
    lo - 1
  }
}

case class RowPartitioner(partitionStarts: Array[Long]) extends Partitioner {
  override val numPartitions: Int = partitionStarts.length - 1

  override def getPartition(key: Any): Int = key match {
    case i: Long => RowPartitioner.findInterval(partitionStarts, i)
  }
}

package is.hail.distributedmatrix

import org.apache.spark.Partitioner

case class RowPartitioner(partitionStarts: Array[Long]) extends Partitioner {
  override val numPartitions: Int = partitionStarts.length - 1

  override def getPartition(key: Any): Int = key match {
    case i: Long => binarySearchBefore(partitionStarts, i)
  }
  
  // a is increasing and may contain duplicate values
  //   returns j - 1 where j is the first index with a(j) < elem
  def binarySearchBefore(a: Array[Long], elem: Long): Int = {
    var low = 1
    var high = a.length - 1
    assert(elem >= a(0) && elem < a(high))
    
    while (low <= high) {
      var mid = (low + high) >>> 1
      val midVal = a(mid)
      
      if (midVal < elem)
        low = mid + 1
      else if (midVal > elem)
        high = mid - 1
      else {
        while (mid < a.length - 2 && a(mid) == a(mid + 1))
          mid += 1
        return mid
      }
    }
    low - 1
  }
}

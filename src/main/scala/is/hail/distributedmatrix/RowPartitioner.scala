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
      val mid = (low + high) >>> 1
      val midVal = a(mid)
      
      if (midVal < elem)
        low = mid + 1
      else if (midVal > elem)
        high = mid - 1
      else {
        var mid0 = mid
        while (mid0 < a.length - 2 && a(mid0) == a(mid0 + 1))
          mid0 += 1
        return mid0
      }
    }
    low - 1
  }
}

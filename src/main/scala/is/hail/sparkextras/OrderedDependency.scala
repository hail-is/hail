package is.hail.sparkextras

import org.apache.spark.NarrowDependency
import org.apache.spark.rdd.RDD

class OrderedDependency[PK, K1, K2, V](p1: OrderedPartitioner[PK, K1], p2: OrderedPartitioner[PK, K2],
  rdd: RDD[(K2, V)]) extends NarrowDependency[(K2, V)](rdd) {
  override def getParents(partitionId: Int): Seq[Int] = {
    val (start, end) = OrderedDependency.getDependencies(p1, p2)(partitionId)
    start until end
  }
}

object OrderedDependency {
  def getDependencies[PK](p1: OrderedPartitioner[PK, _], p2: OrderedPartitioner[PK, _])(partitionId: Int): (Int, Int) = {

    val lastPartition = if (partitionId == p1.rangeBounds.length)
      p2.numPartitions - 1
    else
      p2.getPartitionT(p1.rangeBounds(partitionId))

    if (partitionId == 0)
      (0, lastPartition)
    else {
      val startPartition = p2.getPartitionT(p1.rangeBounds(partitionId - 1))
      (startPartition, lastPartition)
    }
  }
}
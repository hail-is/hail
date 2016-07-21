package org.apache.spark

import org.apache.spark.rdd.RDD

class OrderedDependency[T, K1, K2, V](p1: OrderedPartitioner[T, K1], p2: OrderedPartitioner[T, K2],
  rdd: RDD[(K2, V)]) extends NarrowDependency[(K2, V)](rdd) {
  override def getParents(partitionId: Int): Seq[Int] = {
    val (start, end) = OrderedDependency.getDependencies(p1, p2)(partitionId)
    start until end
  }
}

object OrderedDependency {
  def getDependencies[T](p1: OrderedPartitioner[T, _], p2: OrderedPartitioner[T, _])(partitionId: Int): (Int, Int) = {

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
package is.hail.sparkextras

import is.hail.annotations._
import is.hail.rvd.{OrderedRVD, OrderedRVPartitioner}
import is.hail.utils._
import org.apache.spark._
import org.apache.spark.rdd.RDD

class OrderedDependency2(left: OrderedRVD, right: OrderedRVD) extends NarrowDependency[RegionValue](right.rdd) {
  override def getParents(partitionId: Int): Seq[Int] =
    OrderedDependency2.getDependencies(left.partitioner, right.partitioner)(partitionId)
}

object OrderedDependency2 {
  def getDependencies(p1: OrderedRVPartitioner, p2: OrderedRVPartitioner)(partitionId: Int): Range = {
    val lastPartition = if (partitionId == p1.rangeBounds.length)
      p2.numPartitions - 1
    else
      p2.getPartitionPK(p1.rangeBounds(partitionId))

    if (partitionId == 0)
      0 to lastPartition
    else {
      val startPartition = p2.getPartitionPK(p1.rangeBounds(partitionId - 1))
      startPartition to lastPartition
    }
  }
}

case class OrderedJoinDistinctRDD2Partition(index: Int, leftPartition: Partition, rightPartitions: Array[Partition]) extends Partition

class OrderedJoinDistinctRDD2(left: OrderedRVD, right: OrderedRVD, joinType: String)
  extends RDD[JoinedRegionValue](left.sparkContext,
    Seq[Dependency[_]](new OneToOneDependency(left.rdd),
      new OrderedDependency2(left, right))) {
  assert(joinType == "left" || joinType == "inner")
  override val partitioner: Option[Partitioner] = Some(left.partitioner)

  def getPartitions: Array[Partition] = {
    Array.tabulate[Partition](left.getNumPartitions)(i =>
      OrderedJoinDistinctRDD2Partition(i,
        left.partitions(i),
        OrderedDependency2.getDependencies(left.partitioner, right.partitioner)(i)
          .map(right.partitions)
          .toArray))
  }

  override def getPreferredLocations(split: Partition): Seq[String] = left.rdd.preferredLocations(split)

  override def compute(split: Partition, context: TaskContext): Iterator[JoinedRegionValue] = {
    val partition = split.asInstanceOf[OrderedJoinDistinctRDD2Partition]

    val leftIt = left.rdd.iterator(partition.leftPartition, context)
    val rightIt = partition.rightPartitions.iterator.flatMap { p =>
      right.rdd.iterator(p, context)
    }

    joinType match {
      case "inner" => new OrderedInnerJoinDistinctIterator(left.typ, right.typ, leftIt, rightIt)
      case "left" => new OrderedLeftJoinDistinctIterator(left.typ, right.typ, leftIt, rightIt)
      case _ => fatal(s"Unknown join type `$joinType'. Choose from `inner' or `left'.")
    }
  }
}

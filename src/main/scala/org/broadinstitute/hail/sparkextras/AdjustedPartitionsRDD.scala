package org.broadinstitute.hail.sparkextras

import org.apache.spark.rdd.RDD
import org.apache.spark.{Dependency, NarrowDependency, Partition, TaskContext}

import scala.reflect.ClassTag

case class AdjustedPartitionsRDDPartition[T](index: Int, adjustments: Array[Adjustment[T]]) extends Partition

case class Adjustment[T](index: Int, f: Iterator[T] => Iterator[T])

class AdjustedPartitionsRDD[T](@transient var prev: RDD[T],
  adjustments: IndexedSeq[Array[Adjustment[T]]])(implicit tct: ClassTag[T])
  extends RDD[T](prev.sparkContext, Nil) {
  require(adjustments.length <= prev.partitions.length,
    "invalid adjustments: size greater than previous partitions size")
  require(adjustments.forall(adj => adj.nonEmpty))

  private val parentPartitions = prev.partitions

  override def getPartitions: Array[Partition] = {
    val parentPartitions = dependencies.head.rdd.asInstanceOf[RDD[T]].partitions
    Array.tabulate(adjustments.length) { i =>
      AdjustedPartitionsRDDPartition(i, adjustments(i))
    }
  }

  override def compute(split: Partition, context: TaskContext): Iterator[T] = {
    val parent = dependencies.head.rdd.asInstanceOf[RDD[T]]
    split.asInstanceOf[AdjustedPartitionsRDDPartition[T]]
      .adjustments
      .iterator
      .flatMap { adj =>
        adj.f(parent.compute(parentPartitions(adj.index), context))
      }
  }

  override def getDependencies: Seq[Dependency[_]] = Seq(new NarrowDependency[T](prev) {
    override def getParents(partitionId: Int): Seq[Int] = adjustments(partitionId).map(_.index).toSeq
  })

  override def clearDependencies() {
    super.clearDependencies()
    prev = null
  }

  override def getPreferredLocations(partition: Partition): Seq[String] = {
    val adjustedPartition = partition.asInstanceOf[AdjustedPartitionsRDDPartition[T]]

    val prevPartitions = prev.partitions
    val range = adjustedPartition.adjustments.map(_.index)

    val locationAvail = range.flatMap(i =>
      prev.preferredLocations(prevPartitions(i)))
      .groupBy(identity)
      .mapValues(_.length)

    if (locationAvail.isEmpty)
      return Seq.empty

    val m = locationAvail.values.max
    locationAvail.filter(_._2 == m)
      .keys
      .toSeq
  }
}

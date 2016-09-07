package org.broadinstitute.hail.sparkextras

import org.apache.spark.rdd.RDD
import org.apache.spark.{Dependency, NarrowDependency, Partition, TaskContext}

import scala.reflect.ClassTag

case class AdjustedPartitionsRDDPartition[T](index: Int, parent: Partition, adj: Adjustment[T]) extends Partition

case class Adjustment[T](originalIndex: Int,
  drop: Option[Iterator[T] => Iterator[T]],
  take: Seq[(Int, Iterator[T] => Iterator[T])])

class AdjustedPartitionsRDD[T](@transient var prev: RDD[T], adjustments: IndexedSeq[Adjustment[T]])(implicit tct: ClassTag[T])
  extends RDD[T](prev.sparkContext, Nil) {
  require(adjustments.length <= prev.partitions.length,
    "invalid adjustments: size greater than previous partitions size")
  require(adjustments.forall(adj => adj.take.isEmpty ||
    adj.take.head._1 == adj.originalIndex + 1 && adj.take.zip(adj.take.tail)
      .forall { case ((l, _), (r, _)) => l + 1 == r }),
    "invalid adjustments: nonconsecutive span")

  private val parentPartitions = prev.partitions

  override def getPartitions: Array[Partition] = {
    val parentPartitions = dependencies.head.rdd.asInstanceOf[RDD[T]].partitions
    Array.tabulate(adjustments.length) { i =>
      AdjustedPartitionsRDDPartition(i, parentPartitions(i), adjustments(i))
    }
  }

  override def compute(split: Partition, context: TaskContext): Iterator[T] = {
    val parent = dependencies.head.rdd.asInstanceOf[RDD[T]]
    val adjPartition = split.asInstanceOf[AdjustedPartitionsRDDPartition[T]]
    val adj = adjPartition.adj

    val it = parent.compute(adjPartition.parent, context)
    val dropped: Iterator[T] = adj.drop.map(f => f(it)).getOrElse(it)

    val taken: Iterator[T] = adj.take.iterator
      .flatMap { case (p, f) => f(parent.compute(parentPartitions(p), context)) }

    dropped ++ taken
  }

  override def getDependencies: Seq[Dependency[_]] = Seq(new NarrowDependency[T](prev) {
    override def getParents(partitionId: Int): Seq[Int] = {
      val adj = adjustments(partitionId)
      if (adj.take.isEmpty)
        Seq(adj.originalIndex)
      else
        Seq(adj.originalIndex) ++ adj.take.map(_._1)
    }
  })

  override def clearDependencies() {
    super.clearDependencies()
    prev = null
  }

  override def getPreferredLocations(partition: Partition): Seq[String] = {
    val adjustedPartition = partition.asInstanceOf[AdjustedPartitionsRDDPartition[T]]
    prev.preferredLocations(adjustedPartition.parent) ++ adjustedPartition.adj.take
      .flatMap { case (p, _) => prev.preferredLocations(parentPartitions(p)) }
  }
}

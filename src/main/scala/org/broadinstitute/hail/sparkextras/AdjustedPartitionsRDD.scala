package org.broadinstitute.hail.sparkextras

import org.apache.spark.rdd.RDD
import org.apache.spark.{Dependency, NarrowDependency, Partition, TaskContext}

import scala.reflect.ClassTag

case class AdjustedPartitionsRDDPartition[T](index: Int, parent: Partition, adj: Adjustment[T, Partition]) extends Partition

case class Adjustment[T, U](originalIndex: Int, dropWhile: Option[T => Boolean], takeWhile: Option[(Seq[U], T => Boolean)]) {
  def map[V](f: U => V): Adjustment[T, V] = copy(takeWhile = takeWhile.map { case (it, g) => (it.map(f), g) })
}

class AdjustedPartitionsRDD[T](@transient var prev: RDD[T], adjustments: IndexedSeq[Adjustment[T, Int]])(implicit tct: ClassTag[T])
  extends RDD[T](prev.sparkContext, Nil) {
  require(adjustments.length <= prev.partitions.length, "invalid adjustments: size greater than previous partitions size")
  require(adjustments.forall(adj => adj.takeWhile.forall { case (it, _) =>
    it.nonEmpty && it.head == adj.originalIndex + 1 && it.zip(it.tail).forall { case (l, r) => l + 1 == r }
  }), "invalid adjustments: nonconsecutive span")

  override def getPartitions: Array[Partition] = {
    val parentPartitions = dependencies.head.rdd.asInstanceOf[RDD[T]].partitions
    Array.tabulate(adjustments.length) { i =>
      AdjustedPartitionsRDDPartition(i, parentPartitions(i), adjustments(i).map(parentPartitions))
    }
  }

  override def compute(split: Partition, context: TaskContext): Iterator[T] = {
    val parent = dependencies.head.rdd.asInstanceOf[RDD[T]]
    val adjPartition = split.asInstanceOf[AdjustedPartitionsRDDPartition[T]]
    val adj = adjPartition.adj

    val it = parent.compute(adjPartition.parent, context)
    val dropped: Iterator[T] = adj.dropWhile
      .map { f => it.dropWhile(t => f(t)) }
      .getOrElse(it)

    val taken: Iterator[T] = adj.takeWhile
      .map { case (additionalPartitions, f) =>
        additionalPartitions.iterator
          .flatMap { p => parent.compute(p, context).takeWhile(t => f(t)) }
      }.getOrElse(Iterator())

    dropped ++ taken
  }

  override def getDependencies: Seq[Dependency[_]] = Seq(new NarrowDependency[T](prev) {
    override def getParents(partitionId: Int): Seq[Int] = {
      val adj = adjustments(partitionId)
      adj.takeWhile.map { case (it, _) => Seq(adj.originalIndex) ++ it }
        .getOrElse(Seq(adj.originalIndex))
    }
  })


  override def clearDependencies() {
    super.clearDependencies()
    prev = null
  }

  override def getPreferredLocations(partition: Partition): Seq[String] = {
    val adjustedPartition = partition.asInstanceOf[AdjustedPartitionsRDDPartition[T]]
    prev.preferredLocations(adjustedPartition.parent) ++ adjustedPartition.adj.takeWhile.map { case (parents, _) =>
      parents.flatMap { p => prev.preferredLocations(p) }
    }.getOrElse(Seq())
  }
}

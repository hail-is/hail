package org.broadinstitute.hail.sparkextras

import org.apache.spark._
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

case class GeneralRDDPartition[T](index: Int, parentPartitions: Array[Partition], f: (Array[Iterator[T]]) => Iterator[T]) extends Partition

case class PartitionInput[T](indices: Array[Int], f: (Array[Iterator[T]] => Iterator[T]))

class GeneralRDD[T](@transient var prev: RDD[T],
  @transient val inputs: IndexedSeq[PartitionInput[T]])(implicit tct: ClassTag[T]) extends RDD[T](prev.sparkContext, Nil) {

  override def getPartitions: Array[Partition] = {
    val parentPartitions = prev.partitions
    inputs.zipWithIndex.map { case (input, i) =>
      GeneralRDDPartition(i, input.indices.map(j => parentPartitions(j)), input.f)
    }.toArray
  }

  override def compute(split: Partition, context: TaskContext): Iterator[T] = {
    val parent = dependencies.head.rdd.asInstanceOf[RDD[T]]
    val gp = split.asInstanceOf[GeneralRDDPartition[T]]
    gp.f(gp.parentPartitions.map(p => parent.iterator(p, context)))
  }

  override def getDependencies: Seq[Dependency[_]] = Seq(new NarrowDependency[T](prev) {
    override def getParents(partitionId: Int): Seq[Int] = inputs(partitionId).indices
  })
}
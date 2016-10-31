package org.broadinstitute.hail.sparkextras

import org.apache.spark._
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

case class GeneralRDDPartition[T](index: Int, inputs: Array[(Int, Int)], f: (Array[Iterator[T]]) => Iterator[T]) extends Partition

class GeneralRDD[T](@transient var sc: SparkContext,
  var rdds: Array[RDD[T]],
  var inputs: Array[(Array[(Int, Int)], (Array[Iterator[T]] => Iterator[T]))])(implicit tct: ClassTag[T]) extends RDD[T](sc, Nil) {

  override def getPartitions: Array[Partition] = {
    inputs.zipWithIndex.map { case (input, i) => new GeneralRDDPartition[T](i, input._1, input._2) }
  }

  override def compute(split: Partition, context: TaskContext): Iterator[T] = {
    val gp = split.asInstanceOf[GeneralRDDPartition[T]]
    gp.f(gp.inputs.map { case (rddIndex, partitionIndex) =>
      val rdd = rdds(rddIndex)
      val partition = rdd.partitions(partitionIndex)
      rdd.iterator(partition, context)
    })
  }

  override def getDependencies: Seq[Dependency[_]] = {
    inputs
      .zipWithIndex
      .flatMap { case (input, i) => input._1.map { case (rddIndex, partitionIndex) => (rddIndex, partitionIndex, i) } }
      .groupBy(_._1)
      .map { case (rddIndex, x) =>
        new NarrowDependency[T](rdds(rddIndex)) {
          override def getParents(partitionId: Int): Seq[Int] =
            x.filter {
              partitionId == _._3
            }.map(_._2)
        }
      }.toSeq
  }
}

//case class GeneralRDDPartition[T](index: Int, parentPartitions: Array[Partition], f: (Array[Iterator[T]]) => Iterator[T]) extends Partition
//
//case class PartitionInput[T](indices: Array[Int], f: (Array[Iterator[T]] => Iterator[T]))
//
//class GeneralRDD[T](@transient var prev: RDD[T],
//  @transient val inputs: IndexedSeq[PartitionInput[T]])(implicit tct: ClassTag[T]) extends RDD[T](prev.sparkContext, Nil) {
//
//  override def getPartitions: Array[Partition] = {
//    val parentPartitions = prev.partitions
//    inputs.zipWithIndex.map { case (input, i) =>
//      GeneralRDDPartition(i, input.indices.map(j => parentPartitions(j)), input.f)
//    }.toArray
//  }
//
//  override def compute(split: Partition, context: TaskContext): Iterator[T] = {
//    val parent = dependencies.head.rdd.asInstanceOf[RDD[T]]
//    val gp = split.asInstanceOf[GeneralRDDPartition[T]]
//    gp.f(gp.parentPartitions.map(p => parent.iterator(p, context)))
//  }
//
//  override def getDependencies: Seq[Dependency[_]] = Seq(new NarrowDependency[T](prev) {
//    override def getParents(partitionId: Int): Seq[Int] = inputs(partitionId).indices
//  })
//}
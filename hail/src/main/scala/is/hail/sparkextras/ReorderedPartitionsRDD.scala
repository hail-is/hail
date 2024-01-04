package is.hail.sparkextras

import is.hail.utils.FastSeq

import scala.reflect.ClassTag

import org.apache.spark.{Dependency, NarrowDependency, Partition, TaskContext}
import org.apache.spark.rdd.RDD

case class ReorderedPartitionsRDDPartition(index: Int, oldPartition: Partition) extends Partition

class ReorderedPartitionsRDD[T](
  @transient var prev: RDD[T],
  @transient val oldIndices: Array[Int],
)(implicit tct: ClassTag[T]
) extends RDD[T](prev.sparkContext, Nil) {

  override def getPartitions: Array[Partition] = {
    val parentPartitions = dependencies.head.rdd.asInstanceOf[RDD[T]].partitions
    Array.tabulate(oldIndices.length) { i =>
      val oldIndex = oldIndices(i)
      val oldPartition = parentPartitions(oldIndex)
      ReorderedPartitionsRDDPartition(i, oldPartition)
    }
  }

  override def compute(split: Partition, context: TaskContext): Iterator[T] = {
    val parent = dependencies.head.rdd.asInstanceOf[RDD[T]]
    parent.compute(split.asInstanceOf[ReorderedPartitionsRDDPartition].oldPartition, context)
  }

  override def getDependencies: Seq[Dependency[_]] = FastSeq(new NarrowDependency[T](prev) {
    override def getParents(partitionId: Int): Seq[Int] = FastSeq(oldIndices(partitionId))
  })

  override def clearDependencies() {
    super.clearDependencies()
    prev = null
  }

  override def getPreferredLocations(partition: Partition): Seq[String] =
    prev.preferredLocations(partition.asInstanceOf[ReorderedPartitionsRDDPartition].oldPartition)
}

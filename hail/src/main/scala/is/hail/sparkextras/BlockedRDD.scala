package is.hail.sparkextras

import is.hail.utils._
import org.apache.spark.rdd.RDD
import org.apache.spark.{Dependency, NarrowDependency, Partition, TaskContext}

import scala.language.existentials
import scala.reflect.ClassTag

case class BlockedRDDPartition(@transient rdd: RDD[_],
  index: Int,
  start: Int,
  end: Int) extends Partition {
  require(start <= end)

  val parentPartitions = range.map(rdd.partitions).toArray

  def range: Range = start to end
}

class BlockedRDD[T](@transient var prev: RDD[T],
  @transient val newPartEnd: Array[Int])(implicit tct: ClassTag[T]) extends RDD[T](prev.sparkContext, Nil) {

  override def getPartitions: Array[Partition] = {
    newPartEnd.zipWithIndex.map { case (end, i) =>
      val start = if (i == 0)
        0
      else
        newPartEnd(i - 1) + 1
      BlockedRDDPartition(prev, i, start, end)
    }
  }

  override def compute(split: Partition, context: TaskContext): Iterator[T] = {
    val parent = dependencies.head.rdd.asInstanceOf[RDD[T]]
    split.asInstanceOf[BlockedRDDPartition].parentPartitions.iterator.flatMap(p =>
      parent.iterator(p, context))
  }

  override def getDependencies: Seq[Dependency[_]] = {
    FastSeq(new NarrowDependency(prev) {
      def getParents(id: Int): Seq[Int] =
        partitions(id).asInstanceOf[BlockedRDDPartition].range
    })
  }

  override def clearDependencies() {
    super.clearDependencies()
    prev = null
  }

  override def getPreferredLocations(partition: Partition): Seq[String] = {
    val prevPartitions = prev.partitions
    val range = partition.asInstanceOf[BlockedRDDPartition].range

    val locationAvail = range.flatMap(i =>
      prev.preferredLocations(prevPartitions(i)))
      .groupBy(identity)
      .mapValues(_.length)

    if (locationAvail.isEmpty)
      return FastSeq.empty[String]

    val m = locationAvail.values.max
    locationAvail.filter(_._2 == m)
      .keys
      .toFastSeq
  }
}

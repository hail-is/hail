package is.hail.sparkextras

import is.hail.utils._

import scala.language.existentials
import scala.reflect.ClassTag

import org.apache.spark.{Dependency, NarrowDependency, Partition, TaskContext}
import org.apache.spark.rdd.RDD

case class BlockedRDDPartition(@transient rdd: RDD[_], index: Int, first: Int, last: Int)
    extends Partition {
  require(first <= last)

  val parentPartitions: Array[Partition] = range.map(rdd.partitions).toArray

  def range: Range = first to last
}

class BlockedRDD[T](
  @transient var prev: RDD[T],
  @transient val partFirst: Array[Int],
  @transient val partLast: Array[Int],
)(implicit tct: ClassTag[T]
) extends RDD[T](prev.sparkContext, Nil) {
  assert(partFirst.length == partLast.length)

  override def getPartitions: Array[Partition] =
    Array.tabulate[Partition](partFirst.length)(i =>
      BlockedRDDPartition(prev, i, partFirst(i), partLast(i))
    )

  override def compute(split: Partition, context: TaskContext): Iterator[T] = {
    val parent = dependencies.head.rdd.asInstanceOf[RDD[T]]
    split.asInstanceOf[BlockedRDDPartition].parentPartitions.iterator.flatMap(p =>
      parent.iterator(p, context)
    )
  }

  override def getDependencies: Seq[Dependency[_]] =
    FastSeq(new NarrowDependency(prev) {
      def getParents(id: Int): Seq[Int] =
        partitions(id).asInstanceOf[BlockedRDDPartition].range
    })

  override def clearDependencies() {
    super.clearDependencies()
    prev = null
  }

  override def getPreferredLocations(partition: Partition): Seq[String] = {
    val prevPartitions = prev.partitions
    val range = partition.asInstanceOf[BlockedRDDPartition].range

    val locationAvail = range.flatMap(i =>
      prev.preferredLocations(prevPartitions(i))
    )
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

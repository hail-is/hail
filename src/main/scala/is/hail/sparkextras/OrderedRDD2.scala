package is.hail.sparkextras

import is.hail.annotations._
import is.hail.rvd.{OrderedRVD, OrderedRVDPartitioner, OrderedRVDType}
import is.hail.utils._
import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

class OrderedDependency(prev: OrderedRVD, newPartitioner: OrderedRVDPartitioner)
  extends NarrowDependency[RegionValue](prev.rdd) {

  override def getParents(partitionId: Int): Seq[Int] = {
    val partBounds =
      newPartitioner.rangeBounds(partitionId).asInstanceOf[Interval]
    prev.partitioner.getPartitionRangePK(partBounds)
  }
}

case class RepartitionedOrderedRDD2Partition(
    index: Int,
    parents: Seq[Partition],
    range: Interval)
  extends Partition

class RepartitionedOrderedRDD2(
    prev: OrderedRVD,
    newPartitioner: OrderedRVDPartitioner)
  extends RDD[RegionValue](prev.sparkContext, Nil) { // Nil since we implement getDependencies

  def getPartitions: Array[Partition] = {
    Array.tabulate[Partition](newPartitioner.numPartitions) { i =>
      RepartitionedOrderedRDD2Partition(
        i,
        dependency.getParents(i).map(prev.rdd.partitions),
        newPartitioner.rangeBounds(i).asInstanceOf[Interval])
    }
  }

  override def compute(partition: Partition, context: TaskContext): Iterator[RegionValue] = {
    val ordPartition = partition.asInstanceOf[RepartitionedOrderedRDD2Partition]
    val it = ordPartition.parents.iterator
      .flatMap { parentPartition =>
        prev.rdd.iterator(parentPartition, context)
      }
    OrderedRVIterator(prev.typ, it).restrictToPKInterval(ordPartition.range)
  }

  val dependency = new OrderedDependency(prev, newPartitioner)

  override def getDependencies: Seq[Dependency[_]] = Seq(dependency)
}

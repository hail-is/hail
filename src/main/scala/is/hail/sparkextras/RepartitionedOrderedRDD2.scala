package is.hail.sparkextras

import is.hail.annotations._
import is.hail.rvd.{OrderedRVD, OrderedRVDPartitioner, OrderedRVDType}
import is.hail.utils._
import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

/**
  * Repartition prev to comply with newPartitioner, using narrow dependencies.
  * Assumes new key type is a prefix of old key type, so no reordering is
  * needed. No assumption should need to be made about partition keys, but currently
  * assumes old partition key type is a prefix of the new partition key type.
  */
class RepartitionedOrderedRDD2(
    prev: OrderedRVD,
    newPartitioner: OrderedRVDPartitioner)
  extends RDD[RegionValue](prev.sparkContext, Nil) { // Nil since we implement getDependencies

//  require(newPartitioner.kType isPrefixOf prev.typ.kType)
  // There should really be no precondition on partition keys. Drop this when
  // we're able
  require(prev.typ.pkType isPrefixOf newPartitioner.pkType)

  def getPartitions: Array[Partition] = {
    Array.tabulate[Partition](newPartitioner.numPartitions) { i =>
      RepartitionedOrderedRDD2Partition(
        i,
        dependency.getParents(i).toArray.map(prev.rdd.partitions),
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

class OrderedDependency(prev: OrderedRVD, newPartitioner: OrderedRVDPartitioner)
  extends NarrowDependency[RegionValue](prev.rdd) {

  // no precondition on partition keys
//  require(newPartitioner.kType isPrefixOf prev.typ.kType)
  // There should really be no precondition on partition keys. Drop this when
  // we're able
  require(prev.typ.pkType isPrefixOf newPartitioner.pkType)

  override def getParents(partitionId: Int): Seq[Int] = {
    val partBounds =
      newPartitioner.rangeBounds(partitionId).asInstanceOf[Interval]
    prev.partitioner.getPartitionRange(partBounds)
  }
}

case class RepartitionedOrderedRDD2Partition(
    index: Int,
    parents: Array[Partition],
    range: Interval)
  extends Partition

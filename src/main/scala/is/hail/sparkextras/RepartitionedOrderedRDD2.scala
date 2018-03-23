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
object RepartitionedOrderedRDD2 {
  def apply(prev: OrderedRVD, newPartitioner: OrderedRVDPartitioner): RepartitionedOrderedRDD2 = {
    new RepartitionedOrderedRDD2(prev.rdd, prev.typ, prev.partitioner, newPartitioner)
  }
}

class RepartitionedOrderedRDD2(
    prevRDD: RDD[RegionValue],
    typ: OrderedRVDType,
    oldPartitioner: OrderedRVDPartitioner,
    newPartitioner: OrderedRVDPartitioner)
  extends RDD[RegionValue](prevRDD.sparkContext, Nil) { // Nil since we implement getDependencies

//  require(newPartitioner.kType isPrefixOf prev.typ.kType)
  // There should really be no precondition on partition keys. Drop this when
  // we're able
  require(typ.pkType isPrefixOf newPartitioner.pkType)

  def getPartitions: Array[Partition] = {
    Array.tabulate[Partition](newPartitioner.numPartitions) { i =>
      RepartitionedOrderedRDD2Partition(
        i,
        dependency.getParents(i).toArray.map(prevRDD.partitions),
        newPartitioner.rangeBounds(i).asInstanceOf[Interval])
    }
  }

  override def compute(partition: Partition, context: TaskContext): Iterator[RegionValue] = {
    val ordPartition = partition.asInstanceOf[RepartitionedOrderedRDD2Partition]
    val it = ordPartition.parents.iterator
      .flatMap { parentPartition =>
        prevRDD.iterator(parentPartition, context)
      }
    OrderedRVIterator(typ, it).restrictToPKInterval(ordPartition.range)
  }

  val dependency = new OrderedDependency(oldPartitioner, newPartitioner, prevRDD)

  override def getDependencies: Seq[Dependency[_]] = Seq(dependency)
}

class OrderedDependency(oldPartitioner: OrderedRVDPartitioner, newPartitioner: OrderedRVDPartitioner, rdd: RDD[RegionValue])
  extends NarrowDependency[RegionValue](rdd) {

  // no precondition on partition keys
//  require(newPartitioner.kType isPrefixOf prev.typ.kType)
  // There should really be no precondition on partition keys. Drop this when
  // we're able
  require(oldPartitioner.pkType isPrefixOf newPartitioner.pkType)

  override def getParents(partitionId: Int): Seq[Int] = {
    val partBounds =
      newPartitioner.rangeBounds(partitionId).asInstanceOf[Interval]
    oldPartitioner.getPartitionRange(partBounds)
  }
}

case class RepartitionedOrderedRDD2Partition(
    index: Int,
    parents: Array[Partition],
    range: Interval)
  extends Partition

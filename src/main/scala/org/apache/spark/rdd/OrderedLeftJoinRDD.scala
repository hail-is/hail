package org.apache.spark.rdd

import org.apache.spark._
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.utils.BufferedAdvanceableOrderedPairIterator

import scala.reflect.ClassTag

object OrderedRDDIterator {
  def apply[T, K, V](rdd: OrderedRDD[T, K, V],
    context: TaskContext,
    k0: K)(implicit tOrd: Ordering[T], kOrd: Ordering[K]): OrderedRDDIterator[T, K, V] = {

    val it = new OrderedRDDIterator[T, K, V](rdd, context)
    it.setPartition(rdd.orderedPartitioner.getPartition(k0))
    it
  }
}

class OrderedRDDIterator[T, K, V](
  rdd: OrderedRDD[T, K, V],
  context: TaskContext,
  var partIndex: Int = -1,
  var it: BufferedIterator[(K, V)] = null,
  var partMaxT: T = uninitialized[T])(implicit tOrd: Ordering[T], val kOrdering: Ordering[K]) extends BufferedAdvanceableOrderedPairIterator[K, V] {

  import Ordering.Implicits._

  private val nPartitions = rdd.partitions.length
  private val partitioner = rdd.orderedPartitioner
  private val projectKey = partitioner.projectKey

  def head = it.head

  override def buffered = this

  def hasNext: Boolean = it.hasNext

  def setPartition(newPartIndex: Int) {
    partIndex = newPartIndex
    if (partIndex < nPartitions) {
      it = rdd.iterator(rdd.partitions(partIndex), context).buffered
      if (partIndex < nPartitions - 1)
        partMaxT = partitioner.rangeBounds(partIndex)
    } else
      it = Iterator.empty.buffered
  }

  def next(): (K, V) = {
    val n = it.next()

    /* advance to next partition if necessary */
    while (!it.hasNext && partIndex < nPartitions)
      setPartition(partIndex + 1)

    n
  }

  def advanceTo(k: K) {
    val t = projectKey(k)

    if (partIndex < nPartitions - 1 && t > partMaxT)
      setPartition(partitioner.getPartition(k))

    while (hasNext && head._1 < k)
      next()
  }
}

class OrderedLeftJoinRDD[T, K, V1, V2](left: OrderedRDD[T, K, V1], right: OrderedRDD[T, K, V2])
  (implicit tOrd: Ordering[T], kOrd: Ordering[K], tct: ClassTag[T],
    kct: ClassTag[K]) extends RDD[(K, (V1, Option[V2]))](left.sparkContext, Seq(new OneToOneDependency(left),
  new OrderedDependency(left.orderedPartitioner, right.orderedPartitioner, right)): Seq[Dependency[_]]) {

  override val partitioner: Option[Partitioner] = left.partitioner

  def getPartitions: Array[Partition] = left.partitions

  override def getPreferredLocations(split: Partition): Seq[String] = left.preferredLocations(split)

  override def compute(split: Partition, context: TaskContext): Iterator[(K, (V1, Option[V2]))] = {
    val leftIt = left.iterator(split, context).buffered
    val rightIt = OrderedRDDIterator(right, context, leftIt.head._1)
    leftIt.sortedLeftJoinDistinct(rightIt)
  }
}

package org.broadinstitute.hail.sparkextras

import org.apache.spark.{SparkContext, _}
import org.apache.spark.rdd.{PartitionPruningRDD, RDD, ShuffledRDD}
import org.broadinstitute.hail.Utils._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.util.hashing._

object OrderedRDD {

  def empty[T, K, V](sc: SparkContext, projectKey: (K) => T)(implicit tOrd: Ordering[T], kOrd: Ordering[K], tct: ClassTag[T],
    kct: ClassTag[K]): OrderedRDD[T, K, V] = new OrderedRDD[T, K, V](sc.emptyRDD[(K, V)], OrderedPartitioner.empty[T, K](projectKey))

  def apply[T, K, V](rdd: RDD[(K, V)], projectKey: (K) => T, fastKeys: Option[RDD[K]] = None)
    (implicit tOrd: Ordering[T], kOrd: Ordering[K], tct: ClassTag[T], kct: ClassTag[K], vct: ClassTag[V]): OrderedRDD[T, K, V] = {
    import Ordering.Implicits._

    if (rdd.partitions.isEmpty)
      return empty(rdd.sparkContext, projectKey)

    rdd match {
      case ordd: OrderedRDD[T, K, V] => return ordd
      case _ =>
    }

    rdd.partitioner match {
      case Some(op: OrderedPartitioner[T, K]) => return new OrderedRDD[T, K, V](rdd, op)
      case _ =>
    }

    val keys = fastKeys.getOrElse(rdd.map(_._1))

    val keyInfoOption = anyFailAllFail[Array, PartitionKeyInfo[T]](
      keys.mapPartitionsWithIndex { case (i, it) =>
        Iterator(if (it.hasNext)
          Some(PartitionKeyInfo.apply(i, projectKey, it))
        else
          None)
      }.collect())

    val partitionsSorted =
      keyInfoOption.exists(keyInfo =>
        keyInfo.zip(keyInfo.tail).forall { case (p, pnext) =>
          p.max < pnext.min
        })

    if (partitionsSorted) {
      val keyInfo = keyInfoOption.get
      val partitioner = OrderedPartitioner[T, K](keyInfo.init.map(pi => pi.max), projectKey)
      val sortedness = keyInfo.map(_.sortedness).min
      (sortedness: @unchecked) match {
        case PartitionKeyInfo.KSORTED =>
          assert(sortedness == PartitionKeyInfo.KSORTED)
          info("Coerced sorted dataset")
          new OrderedRDD(rdd, partitioner)

        case PartitionKeyInfo.TSORTED =>
          info("Coerced almost-sorted dataset")
          new OrderedRDD(rdd.mapPartitions { it =>
            it.localKeySort(projectKey)
          }, partitioner)

        case PartitionKeyInfo.UNSORTED =>
          info("Coerced unsorted dataset")
          new OrderedRDD(rdd.mapPartitions { it =>
            it.toArray.sortBy(_._1).iterator
          }, partitioner)
      }
    } else {
      info("Ordering unsorted dataset with network shuffle")
      val ranges: Array[T] = calculateKeyRanges[T](keys.map(projectKey))
      val partitioner = OrderedPartitioner[T, K](ranges, projectKey)
      new OrderedRDD[T, K, V](new ShuffledRDD[K, V, V](rdd, partitioner).setKeyOrdering(kOrd), partitioner)
    }
  }

  /**
    * Copied from:
    *   org.apache.spark.RangePartitioner
    * version 1.5.0
    */
  def calculateKeyRanges[T](rdd: RDD[T])(implicit ord: Ordering[T], tct: ClassTag[T]): Array[T] = {
    // This is the sample size we need to have roughly balanced output partitions, capped at 1M.
    val sampleSize = math.min(20.0 * rdd.partitions.length, 1e6)
    // Assume the input partitions are roughly balanced and over-sample a little bit.
    val sampleSizePerPartition = math.ceil(3.0 * sampleSize / rdd.partitions.length).toInt
    val (numItems, sketched) = OrderedPartitioner.sketch(rdd, sampleSizePerPartition)
    if (numItems == 0L) {
      Array.empty
    } else {
      // If a partition contains much more than the average number of items, we re-sample from it
      // to ensure that enough items are collected from that partition.
      val fraction = math.min(sampleSize / math.max(numItems, 1L), 1.0)
      val candidates = ArrayBuffer.empty[(T, Float)]
      val imbalancedPartitions = mutable.Set.empty[Int]
      sketched.foreach {
        case (idx, n, sample) =>
          if (fraction * n > sampleSizePerPartition) {
            imbalancedPartitions += idx
          } else {
            // The weight is 1 over the sampling probability.
            val weight = (n.toDouble / sample.length).toFloat
            for (key <- sample) {
              candidates += ((key, weight))
            }
          }
      }
      if (imbalancedPartitions.nonEmpty) {
        // Re-sample imbalanced partitions with the desired sampling probability.
        val imbalanced = new PartitionPruningRDD(rdd, imbalancedPartitions.contains)
        val seed = byteswap32(-rdd.id - 1)
        val reSampled = imbalanced.sample(withReplacement = false, fraction, seed).collect()
        val weight = (1.0 / fraction).toFloat
        candidates ++= reSampled.map(x => (x, weight))
      }
      OrderedPartitioner.determineBounds(candidates, rdd.partitions.length)
    }
  }

}

case class BlockedRDDPartition(index: Int,
  start: Int,
  end: Int) extends Partition {
  require(start <= end)

  def range: Range = start to end
}

class BlockedRDD[T](rdd: RDD[T],
  newPartEnd: Array[Int])(implicit tct: ClassTag[T]) extends RDD[T](rdd) {

  override def getPartitions: Array[Partition] = {
    newPartEnd.zipWithIndex.map { case (end, i) =>
      val start = if (i == 0)
        0
      else
        newPartEnd(i - 1) + 1
      BlockedRDDPartition(i, start, end)
    }
  }

  override def compute(split: Partition, context: TaskContext): Iterator[T] = {
    split.asInstanceOf[BlockedRDDPartition].range.iterator.flatMap { i =>
      rdd.iterator(new Partition {
        def index = i
      }, context)
    }
  }
}

class OrderedRDD[T, K, V](rdd: RDD[(K, V)],
  val orderedPartitioner: OrderedPartitioner[T, K])
  (implicit tOrd: Ordering[T], kOrd: Ordering[K], tct: ClassTag[T], kct: ClassTag[K]) extends RDD[(K, V)](rdd) {
  assert((orderedPartitioner.rangeBounds.isEmpty && rdd.partitions.isEmpty)
    || orderedPartitioner.rangeBounds.length == rdd.partitions.length - 1)

  override val partitioner: Option[Partitioner] = Some(orderedPartitioner)

  val getPartitions: Array[Partition] = rdd.partitions

  override def compute(split: Partition, context: TaskContext): Iterator[(K, V)] = rdd.iterator(split, context)

  override def getPreferredLocations(split: Partition): Seq[String] = rdd.preferredLocations(split)

  def orderedLeftJoinDistinct[V2](other: OrderedRDD[T, K, V2]): RDD[(K, (V, Option[V2]))] =
    new OrderedLeftJoinRDD[T, K, V, V2](this, other)

  def mapMonotonic[K2, V2](f: (K, V) => (K2, V2), projectKey2: (K2) => T)(implicit k2Ord: Ordering[K2], k2ct: ClassTag[K2]): OrderedRDD[T, K2, V2] = {
    new OrderedRDD[T, K2, V2](
      rdd.mapPartitions(_.map(f.tupled)),
      orderedPartitioner.mapMonotonic(projectKey2))
  }
}
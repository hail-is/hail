package org.broadinstitute.hail.sparkextras

import java.util

import org.apache.spark.rdd.{PartitionPruningRDD, RDD, ShuffledRDD}
import org.apache.spark.storage.StorageLevel
import org.broadinstitute.hail.utils._
import org.apache.spark.{SparkContext, _}
import org.broadinstitute.hail.variant.Variant

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.language.existentials
import scala.reflect.ClassTag
import scala.util.hashing._

object OrderedRDD {

  type CoercionMethod = Int

  final val ORDERED_PARTITIONER: CoercionMethod = 0
  final val AS_IS: CoercionMethod = 1
  final val LOCAL_SORT: CoercionMethod = 2
  final val ARRAY_SORT: CoercionMethod = 3
  final val SHUFFLE: CoercionMethod = 4

  def empty[PK, K, V](sc: SparkContext)(implicit kOk: OrderedKey[PK, K], vct: ClassTag[V]): OrderedRDD[PK, K, V] =
    new OrderedRDD[PK, K, V](sc.emptyRDD[(K, V)], OrderedPartitioner.empty)

  def shuffle[PK, K, V](rdd: RDD[(K, V)], partitioner: OrderedPartitioner[PK, K])
    (implicit kOk: OrderedKey[PK, K], vct: ClassTag[V]): OrderedRDD[PK, K, V] = {
    import kOk._
    OrderedRDD[PK, K, V](new ShuffledRDD[K, V, V](rdd, partitioner).setKeyOrdering(kOk.kOrd), partitioner)
  }

  def cast[PK, K, V](rdd: RDD[(K, V)])(implicit kOk: OrderedKey[PK, K], vct: ClassTag[V]): OrderedRDD[PK, K, V] = {
    if (rdd.partitions.isEmpty)
      OrderedRDD.empty[PK, K, V](rdd.sparkContext)
    else
      rdd match {
        case ordered: OrderedRDD[PK, K, V] => ordered
        case _ =>
          (rdd.partitioner: @unchecked) match {
            case Some(p: OrderedPartitioner[PK, K]) => OrderedRDD(rdd, p)
          }
      }
  }

  def apply[PK, K, V](rdd: RDD[(K, V)], fastKeys: Option[RDD[K]], hintPartitioner: Option[OrderedPartitioner[PK, K]])
    (implicit kOk: OrderedKey[PK, K], vct: ClassTag[V]): OrderedRDD[PK, K, V] = {
    val (_, orderedRDD) = coerce(rdd, fastKeys, hintPartitioner)
    orderedRDD
  }

  def coerce[PK, K, V](rdd: RDD[(K, V)], fastKeys: Option[RDD[K]] = None, hintPartitioner: Option[OrderedPartitioner[PK, K]] = None)
    (implicit kOk: OrderedKey[PK, K], vct: ClassTag[V]): (CoercionMethod, OrderedRDD[PK, K, V]) = {
    import kOk._

    import Ordering.Implicits._

    if (rdd.partitions.isEmpty)
      return (ORDERED_PARTITIONER, empty(rdd.sparkContext))

    rdd match {
      case ordd: OrderedRDD[PK, K, V] => return (ORDERED_PARTITIONER, ordd)
      case _ =>
    }

    rdd.partitioner match {
      case Some(op: OrderedPartitioner[PK, K]) => return (ORDERED_PARTITIONER, new OrderedRDD[PK, K, V](rdd, op))
      case _ =>
    }

    val keys = fastKeys.getOrElse(rdd.map(_._1))

    val keyInfo = keys.mapPartitionsWithIndex { case (i, it) =>
      if (it.hasNext)
        Iterator(PartitionKeyInfo(i, it))
      else
        Iterator()
    }.collect()

    log.info(s"keyInfo = ${ keyInfo.toSeq }")

    if (keyInfo.isEmpty)
      return (AS_IS, empty(rdd.sparkContext))

    val sortedKeyInfo = keyInfo.sortBy(_.min)
    val partitionsSorted = sortedKeyInfo.zip(sortedKeyInfo.tail).forall { case (p, pnext) =>
      val r = p.max < pnext.min
      if (!r)
        log.info(s"not sorted: p = $p, pnext = $pnext")
      r
    }

    if (partitionsSorted) {
      val partitioner = OrderedPartitioner[PK, K](sortedKeyInfo.init.map(_.max), sortedKeyInfo.length)
      val sortedness = sortedKeyInfo.map(_.sortedness).min
      val reorderedPartitionsRDD = rdd.reorderPartitions(sortedKeyInfo.map(_.partIndex))
      (sortedness: @unchecked) match {
        case PartitionKeyInfo.KSORTED =>
          assert(sortedness == PartitionKeyInfo.KSORTED)
          info("Coerced sorted dataset")
          (AS_IS, OrderedRDD(reorderedPartitionsRDD, partitioner))

        case PartitionKeyInfo.TSORTED =>
          info("Coerced almost-sorted dataset")
          (LOCAL_SORT, OrderedRDD(reorderedPartitionsRDD.mapPartitions { it =>
            localKeySort(it)
          }, partitioner))

        case PartitionKeyInfo.UNSORTED =>
          info("Coerced unsorted dataset")
          (ARRAY_SORT, OrderedRDD(reorderedPartitionsRDD.mapPartitions { it =>
            it.toArray.sortBy(_._1).iterator
          }, partitioner))
      }
    } else {
      info("Ordering unsorted dataset with network shuffle")
      val p = hintPartitioner
        .filter(_.numPartitions >= rdd.partitions.length)
        .getOrElse {
          val ranges = calculateKeyRanges(keys.map(kOk.project))
          OrderedPartitioner(ranges, ranges.length + 1)
        }
      (SHUFFLE, shuffle(rdd, p))
    }
  }
  
  def apply[PK, K, V](rdd: RDD[(K, V)],
    orderedPartitioner: OrderedPartitioner[PK, K])
    (implicit kOk: OrderedKey[PK, K], vct: ClassTag[V]): OrderedRDD[PK, K, V] = {
    import kOk._

    import Ordering.Implicits._

    rdd.partitioner match {
      /* if we verified rdd is K-sorted, it won't necessarily be partitioned */
      case Some(p) => assert(p eq orderedPartitioner)
      case _ =>
    }

    val rangeBoundsBc = rdd.sparkContext.broadcast(orderedPartitioner.rangeBounds)
    new OrderedRDD(
      rdd.mapPartitionsWithIndex { case (i, it) =>
        new Iterator[(K, V)] {
          var prevK: K = _
          var first = true

          def hasNext = it.hasNext

          def next(): (K, V) = {
            val r = it.next()

            if (i < rangeBoundsBc.value.length)
              assert(kOk.project(r._1) <= rangeBoundsBc.value(i))
            if (i > 0)
              assert(rangeBoundsBc.value(i - 1) < kOk.project(r._1),
                s"key ${ r._1 } >= last max ${ rangeBoundsBc.value(i - 1) } in partition $i")

            if (first)
              first = false
            else
              assert(prevK <= r._1)

            prevK = r._1
            r
          }
        }
      }, orderedPartitioner)
  }

  /**
    * Precondition: the iterator it is T-sorted. Moreover, projectKey must be monotonic. We lazily K-sort each block
    * of T-equivalent elements.
    */
  def localKeySort[PK, K, V](it: Iterator[(K, V)])(implicit kOk: OrderedKey[PK, K]): Iterator[(K, V)] = {
    implicit val kvOrd = new Ordering[(K, V)] {
      // ascending
      def compare(x: (K, V), y: (K, V)): Int = -kOk.kOrd.compare(x._1, y._1)
    }

    val bit = it.buffered

    new Iterator[(K, V)] {
      val q = new mutable.PriorityQueue[(K, V)]

      def hasNext = bit.hasNext || q.nonEmpty

      def next() = {
        if (q.isEmpty) {
          val kv = bit.next()
          val t = kOk.project(kv._1)

          q.enqueue(kv)

          while (bit.hasNext && kOk.project(bit.head._1) == t)
            q.enqueue(bit.next())
        }

        q.dequeue()
      }
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

class OrderedRDD[PK, K, V] private(rdd: RDD[(K, V)], val orderedPartitioner: OrderedPartitioner[PK, K])
  extends RDD[(K, V)](rdd) {
  implicit val kOk: OrderedKey[PK, K] = orderedPartitioner.kOk

  import kOk._

  log.info(s"partitions: ${ rdd.partitions.length }, ${ orderedPartitioner.rangeBounds.length }")

  assert(orderedPartitioner.numPartitions == rdd.partitions.length)

  override val partitioner: Option[Partitioner] = Some(orderedPartitioner)

  override def getPartitions: Array[Partition] = rdd.partitions

  override def compute(split: Partition, context: TaskContext): Iterator[(K, V)] = rdd.iterator(split, context)

  override def getPreferredLocations(split: Partition): Seq[String] = rdd.preferredLocations(split)

  def orderedLeftJoinDistinct[V2](other: OrderedRDD[PK, K, V2]): RDD[(K, (V, Option[V2]))] =
    new OrderedLeftJoinRDD[PK, K, V, V2](this, other)

  def mapMonotonic[K2, V2](mapKey: OrderedKeyFunction[PK, K, PK, K2], mapValue: (K, V) => (V2)): OrderedRDD[PK, K2, V2] = {
    new OrderedRDD[PK, K2, V2](rdd.mapPartitions(_.map { case (k, v) => (mapKey(k), mapValue(k, v)) }),
      orderedPartitioner.mapMonotonic(mapKey.k2ok))
  }

  /**
    * Preconditions:
    *
    * - if v1 < v2, f(v1) = (v1a, v1b, ...) and f(v2) = (v2p, v2q, ...), then v1x < v1y for all x = a, b, ... and y = p, q, ...
    *
    * - for all x, kOk.project(v1) = kOk.project(v1x)
    *
    * - the TraversableOnce is sorted according to kOk.kOrd
    */
  def flatMapMonotonic[V2](f: (K, V) => TraversableOnce[(K, V2)]): OrderedRDD[PK, K, V2] = {
    new OrderedRDD[PK, K, V2](rdd.mapPartitions(_.flatMap(f.tupled)), orderedPartitioner)
  }

  /**
    * Preconditions:
    *
    * - if v1 < v2, f(v1) = (v1a, v1b, ...) and f(v2) = (v2p, v2q, ...), then v1x < v1y for all x = a, b, ... and y = p, q, ...
    *
    * - for all x, kOk.project(v1) = kOk.project(v1x) (and similarly for v2)
    *
    * - the TraversableOnce is sorted according to k2Ok.kOrd
    */
  def flatMapMonotonic[K2, V2](f: (K, V) => TraversableOnce[(K2, V2)])
    (implicit k2Ok: OrderedKey[PK, K2]): OrderedRDD[PK, K2, V2] = {
    new OrderedRDD[PK, K2, V2](rdd.mapPartitions(_.flatMap(f.tupled)), orderedPartitioner.mapMonotonic)
  }

  override def coalesce(maxPartitions: Int, shuffle: Boolean = false)(implicit ord: Ordering[(K, V)] = null): RDD[(K, V)] = {
    require(maxPartitions > 0, "cannot coalesce to nPartitions <= 0")
    if (shuffle)
      return super.coalesce(maxPartitions, shuffle)(ord)

    val n = rdd.partitions.length
    if (maxPartitions >= n)
      return this

    val persisted = rdd.persist(StorageLevel.MEMORY_AND_DISK)

    val partSize = new Array[Int](n)
    persisted.mapPartitionsWithIndex((i, it) => Iterator((i, it.length)))
      .collect()
      .foreach { case (i, size) => partSize(i) = size }

    val partCommulativeSize = mapAccumulate[Array, Int, Int, Int](partSize, 0)((s, acc) => (s + acc, s + acc))
    val totalSize = partCommulativeSize.last

    var newPartEnd = (0 until maxPartitions).map { i =>
      val t = totalSize * (i + 1) / maxPartitions

      /* j largest index not greater than t */
      var j = util.Arrays.binarySearch(partCommulativeSize, t)
      if (j < 0)
        j = -j - 1
      while (j < partCommulativeSize.length - 1
        && partCommulativeSize(j + 1) == t)
        j += 1
      assert(t <= partCommulativeSize(j) &&
        (j == partCommulativeSize.length - 1 ||
          t < partCommulativeSize(j + 1)))
      j
    }.toArray

    newPartEnd = newPartEnd.zipWithIndex.filter { case (end, i) => i == 0 || newPartEnd(i) != newPartEnd(i - 1) }
      .map(_._1)

    assert(newPartEnd.last == n - 1)
    assert(newPartEnd.zip(newPartEnd.tail).forall { case (i, inext) => i < inext })

    if (newPartEnd.length < maxPartitions)
      warn(s"coalesced to ${ newPartEnd.length } partitions, less than requested $maxPartitions")

    val newRangeBounds = newPartEnd.init.map(orderedPartitioner.rangeBounds)
    val partitioner = new OrderedPartitioner(newRangeBounds, newPartEnd.length)
    new OrderedRDD[PK, K, V](new BlockedRDD(persisted, newPartEnd), partitioner)
  }
}

trait OrderedKey[PK, K] extends Serializable {
  /**
    * This method must be monotonic in {@code k}
    */
  def project(key: K): PK

  implicit def kOrd: Ordering[K]

  implicit def pkOrd: Ordering[PK]

  implicit def kct: ClassTag[K]

  implicit def pkct: ClassTag[PK]
}


abstract class OrderedKeyFunction[PK1, K1, PK2, K2]
(implicit val k1ok: OrderedKey[PK1, K1], val k2ok: OrderedKey[PK2, K2]) extends Serializable {
  def apply(k1: K1): K2 = f(k1)

  /**
    * This method must be monotonic in {@code k1}
    */
  def f(k1: K1): K2

  /**
    * This method must be monotonic in {@code pk1} and must commute with {@code f}:
    *
    * {@code partitionF ∘ k1ok.project = k2ok.project ∘ f}
    *
    * <pre>
    * K1 -------------- f ------------> K2
    * |                                |
    * V                                V
    * K1.PartitionKey -- partitionF -> K2.PartitionKey
    * </pre>
    *
    */
  def partitionF(pk1: PK1): PK2
}

object OrderedKeyFunction {
  def apply[PK, K1, K2](g: (K1) => K2)
    (implicit k1ok: OrderedKey[PK, K1], k2ok: OrderedKey[PK, K2]): OrderedKeyFunction[PK, K1, PK, K2] = {
    OrderedKeyFunction(g, identity)
  }

  def apply[K1, PK1, K2, PK2](g: (K1) => K2, partitionG: (PK1) => PK2)
    (implicit k1ok: OrderedKey[PK1, K1], k2ok: OrderedKey[PK2, K2]): OrderedKeyFunction[PK1, K1, PK2, K2] = {
    new OrderedKeyFunction[PK1, K1, PK2, K2] {
      def f(k1: K1) = g(k1);

      def partitionF(pk1: PK1) = partitionG(pk1)
    }
  }
}
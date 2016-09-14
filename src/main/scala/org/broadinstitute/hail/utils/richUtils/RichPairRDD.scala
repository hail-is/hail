package org.broadinstitute.hail.utils.richUtils

import org.apache.spark.Partitioner
import org.apache.spark.Partitioner._
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.sparkextras.{OrderedKey, OrderedPartitioner, OrderedRDD}
import org.broadinstitute.hail.utils.SpanningIterator

import scala.collection.TraversableOnce
import scala.reflect.ClassTag

class RichPairRDD[K, V](val rdd: RDD[(K, V)]) extends AnyVal {

  def forall(p: ((K, V)) => Boolean)(implicit kct: ClassTag[K], vct: ClassTag[V]): Boolean = rdd.map(p).fold(true)(_ && _)

  def exists(p: ((K, V)) => Boolean)(implicit kct: ClassTag[K], vct: ClassTag[V]): Boolean = rdd.map(p).fold(false)(_ || _)

  def mapValuesWithKey[W](f: (K, V) => W): RDD[(K, W)] = rdd.mapPartitions(_.map { case (k, v) => (k, f(k, v)) },
    preservesPartitioning = true)

  def flatMapValuesWithKey[W](f: (K, V) => TraversableOnce[W]): RDD[(K, W)] = rdd.mapPartitions(_.flatMap { case (k, v) =>
    f(k, v).map(w => (k, w))
  }, preservesPartitioning = true)

  def spanByKey()(implicit kct: ClassTag[K], vct: ClassTag[V]): RDD[(K, Iterable[V])] =
    rdd.mapPartitions(p => new SpanningIterator(p))

  def leftOuterJoinDistinct[W](other: RDD[(K, W)])
    (implicit kt: ClassTag[K], vt: ClassTag[V], ord: Ordering[K] = null): RDD[(K, (V, Option[W]))] = leftOuterJoinDistinct(other, defaultPartitioner(rdd, other))

  def leftOuterJoinDistinct[W](other: RDD[(K, W)], partitioner: Partitioner)
    (implicit kt: ClassTag[K], vt: ClassTag[V], ord: Ordering[K] = null) = {
    rdd.cogroup(other, partitioner).flatMapValues { pair =>
      val w = pair._2.headOption
      pair._1.map((_, w))
    }
  }

  def asOrderedRDD[PK](implicit kOk: OrderedKey[PK, K], vct: ClassTag[V]): OrderedRDD[PK, K, V] =
    OrderedRDD.cast[PK, K, V](rdd)

  def toOrderedRDD[PK](partitioner: OrderedPartitioner[PK, K])(implicit kOk: OrderedKey[PK, K], vct: ClassTag[V]): OrderedRDD[PK, K, V] = {
    OrderedRDD.shuffle(rdd, partitioner)
  }

  def toOrderedRDD[PK](reducedRepresentation: RDD[K])
    (implicit kOk: OrderedKey[PK, K], vct: ClassTag[V]): OrderedRDD[PK, K, V] =
    OrderedRDD[PK, K, V](rdd, Some(reducedRepresentation))

  def toOrderedRDD[PK](implicit kOk: OrderedKey[PK, K], vct: ClassTag[V]): OrderedRDD[PK, K, V] =
    OrderedRDD[PK, K, V](rdd, None)
}

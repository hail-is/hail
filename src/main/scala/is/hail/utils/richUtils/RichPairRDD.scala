package is.hail.utils.richUtils

import org.apache.spark.Partitioner
import org.apache.spark.Partitioner._
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

class RichPairRDD[K, V](val rdd: RDD[(K, V)]) extends AnyVal {

  def forall(p: ((K, V)) => Boolean)(implicit kct: ClassTag[K], vct: ClassTag[V]): Boolean = rdd.map(p).fold(true)(_ && _)

  def exists(p: ((K, V)) => Boolean)(implicit kct: ClassTag[K], vct: ClassTag[V]): Boolean = rdd.map(p).fold(false)(_ || _)

  def mapValuesWithKey[W](f: (K, V) => W): RDD[(K, W)] = rdd.mapPartitions(_.map { case (k, v) => (k, f(k, v)) },
    preservesPartitioning = true)

  def leftOuterJoinDistinct[W](other: RDD[(K, W)])
    (implicit kt: ClassTag[K], vt: ClassTag[V]): RDD[(K, (V, Option[W]))] = leftOuterJoinDistinct(other, defaultPartitioner(rdd, other))

  def leftOuterJoinDistinct[W](other: RDD[(K, W)], partitioner: Partitioner)
    (implicit kt: ClassTag[K], vt: ClassTag[V]) = {
    rdd.cogroup(other, partitioner).flatMapValues { pair =>
      val w = pair._2.headOption
      pair._1.map((_, w))
    }
  }
}
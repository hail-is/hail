package is.hail.sparkextras

import org.apache.spark._
import org.apache.spark.rdd._

import scala.reflect.ClassTag

class ContextPairRDDFunctions[C <: AutoCloseable, K: ClassTag, V: ClassTag](
  crdd: ContextRDD[C, (K, V)]
) {
  def shuffle(p: Partitioner, o: Ordering[K]): ContextRDD[C, (K, V)] =
    // NB: the run marks the end of a context lifetime, the next context
    // lifetime starts after the shuffle (and potentially on a different machine
    // than the one that owned the previous lifetime)
    ContextRDD.weaken(
      new ShuffledRDD[K, V, V](crdd.run, p).setKeyOrdering(o),
      crdd.mkc)

  // FIXME: this needs to take RDDs with values that are not in region-value-land.
  def partitionBy(p: Partitioner): ContextRDD[C, (K, V)] =
    if (crdd.partitioner.contains(p)) crdd
    else ContextRDD.weaken(
      new ShuffledRDD[K, V, V](crdd.run, p),
      crdd.mkc)

  def values: ContextRDD[C, V] = crdd.map(_._2)
}

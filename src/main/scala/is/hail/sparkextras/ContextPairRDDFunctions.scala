package is.hail.sparkextras

import org.apache.spark._
import org.apache.spark.rdd._

import scala.reflect.ClassTag

class ContextPairRDDFunctions[C <: ResettableContext, K: ClassTag, V: ClassTag](
  crdd: ContextRDD[C, (K, V)]
) {
  def shuffle(p: Partitioner, o: Ordering[K]): ContextRDD[C, (K, V)] =
    // NB: the run marks the end of a context lifetime, the next context
    // lifetime starts after the shuffle (and potentially on a different machine
    // than the one that owned the previous lifetime)
    ContextRDD.weaken(
      new ShuffledRDD[K, V, V](crdd.run, p).setKeyOrdering(o),
      crdd.mkc)
}

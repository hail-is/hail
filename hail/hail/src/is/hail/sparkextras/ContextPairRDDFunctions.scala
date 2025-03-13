package is.hail.sparkextras

import scala.reflect.ClassTag

import org.apache.spark._
import org.apache.spark.rdd._

class ContextPairRDDFunctions[K: ClassTag, V: ClassTag](
  crdd: ContextRDD[(K, V)]
) {
  // FIXME: this needs to take RDDs with values that are not in region-value-land.
  def partitionBy(p: Partitioner): ContextRDD[(K, V)] =
    if (crdd.partitioner.contains(p)) crdd
    else ContextRDD.weaken(
      new ShuffledRDD[K, V, V](crdd.run, p)
    )

  def values: ContextRDD[V] = crdd.map(_._2)
}

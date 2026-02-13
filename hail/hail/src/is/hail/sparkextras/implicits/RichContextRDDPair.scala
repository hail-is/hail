package is.hail.sparkextras.implicits

import is.hail.sparkextras.ContextRDD

import scala.reflect.ClassTag

import org.apache.spark._
import org.apache.spark.rdd._

class RichContextRDDPair[K, V](val crdd: ContextRDD[(K, V)]) extends AnyVal {
  // FIXME: this needs to take RDDs with values that are not in region-value-land.
  def partitionBy(p: Partitioner)(implicit KT: ClassTag[K], VT: ClassTag[V]): ContextRDD[(K, V)] =
    if (crdd.partitioner.contains(p)) crdd
    else ContextRDD.weaken(new ShuffledRDD[K, V, V](crdd.run, p))

  def values(implicit VT: ClassTag[V]): ContextRDD[V] =
    crdd.map(_._2)
}

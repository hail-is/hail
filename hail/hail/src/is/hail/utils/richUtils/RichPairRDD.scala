package is.hail.utils.richUtils

import org.apache.spark.rdd.RDD

class RichPairRDD[K, V](val rdd: RDD[(K, V)]) extends AnyVal {
  def mapValuesWithKey[W](f: (K, V) => W): RDD[(K, W)] =
    rdd.mapPartitions(_.map { case (k, v) => (k, f(k, v)) }, preservesPartitioning = true)
}

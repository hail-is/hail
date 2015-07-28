package org.broadinstitute.k3.methods

import scala.collection.Map
import org.apache.spark.rdd.RDD
import org.broadinstitute.k3.variant.{Genotype, Variant}

object SampleNoCall {
  def apply(rdd: RDD[((String, Variant), Genotype)]): Map[String, Int] =
    rdd
      .map({ case ((s, v), g) => (s, if (g.notCalled) 1 else 0) })
      .reduceByKey(_ + _)
      .collectAsMap()
}

package org.broadinstitute.k3.methods

import scala.collection.Map
import org.apache.spark.rdd.RDD
import org.broadinstitute.k3.variant._

object SampleNoCall {
  def apply(vds: VariantDataset): Map[Int, Int] = {
    vds
      .mapValues(g => if (g.notCalled) 1 else 0)
      .reduceBySample(_ + _)
  }
}

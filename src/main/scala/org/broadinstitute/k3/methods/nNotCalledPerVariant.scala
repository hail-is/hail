package org.broadinstitute.k3.methods

import org.apache.spark.rdd.RDD
import org.broadinstitute.k3.variant._

object nNotCalledPerVariant extends VariantMethod[Int] {
  def name = "nNotCalled"

  def apply(vds: VariantDataset): RDD[(Variant, Int)] = {
    vds
      .mapValues(g => if (g.isNotCalled) 1 else 0)
      .reduceByVariant(_ + _)
  }
}

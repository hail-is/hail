package org.broadinstitute.k3.methods

import org.apache.spark.rdd.RDD
import org.broadinstitute.k3.variant._

object nHetPerVariant extends VariantMethod[Int] {
  def name = "nHet"

  def apply(vds: VariantDataset): RDD[(Variant, Int)] = {
    vds
      .mapValues(g => if (g.isHet) 1 else 0)
      .reduceByVariant(_ + _)
  }
}

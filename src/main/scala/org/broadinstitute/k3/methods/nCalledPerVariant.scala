package org.broadinstitute.k3.methods

import org.apache.spark.rdd.RDD
import org.broadinstitute.k3.variant._

object nCalledPerVariant extends VariantMethod[Int] {
  def name = "nCalled"

  def apply(vds: VariantDataset): RDD[(Variant, Int)] = {
    vds
      .mapValues(g => if (g.isCalled) 1 else 0)
      .foldByVariant(0)(_ + _)
  }
}

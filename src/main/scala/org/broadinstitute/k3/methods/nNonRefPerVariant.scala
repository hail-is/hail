package org.broadinstitute.k3.methods

import org.apache.spark.rdd.RDD
import org.broadinstitute.k3.variant._

object nNonRefPerVariant extends VariantMethod[Int] {
  def name = "nNonRef"

  def apply(vds: VariantDataset): RDD[(Variant, Int)] = {
    nGenotypeVectorPerVariant(vds)
      .mapValues(a => a._2 + a._3)
  }
}

package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

import scala.collection.Map

object nHetPerVariant {
  def apply(vds: VariantDataset): Map[Variant, Int] = {
    vds
      .mapValues(g => if (g.isHet) 1 else 0)
      .reduceByVariant(_ + _)
  }
}

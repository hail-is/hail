package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

import scala.collection.Map

object nHomVarPerVariant {
  def apply(vds: VariantDataset): Map[Variant, Int] = {
    vds
      .mapValues(g => if (g.isHomVar) 1 else 0)
      .reduceByVariant(_ + _)
  }
}

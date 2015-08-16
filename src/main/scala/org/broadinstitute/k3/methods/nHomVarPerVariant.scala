package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

object nHomVarPerVariant extends VariantMethod[Int] {
  def name = "nHomVar"

  def apply(vds: VariantDataset): Map[Variant, Int] = {
    vds
      .mapValues(g => if (g.isHomVar) 1 else 0)
      .reduceByVariant(_ + _)
  }
}

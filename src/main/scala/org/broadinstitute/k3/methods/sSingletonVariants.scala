package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

object sSingletonVariants {
  def apply(vds: VariantDataset): Set[Variant] = {
    vds
      .mapValues(g => if (g.isNonRef) 1 else 0)
      .foldByVariant(0)(_ + _)
      .filter(_._2 == 1)
      .keys
      .collect().toSet
  }
}

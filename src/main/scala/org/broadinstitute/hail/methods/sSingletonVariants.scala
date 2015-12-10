package org.broadinstitute.hail.methods

import org.broadinstitute.hail.variant._

object sSingletonVariants {
  def apply(vds: VariantDataset): Set[Variant] = {
    vds
      .mapValues(g => if (g.isCalledNonRef) 1 else 0)
      .foldByVariant(0)(_ + _)
      .filter(_._2 == 1)
      .keys
      .collect().toSet
  }
}

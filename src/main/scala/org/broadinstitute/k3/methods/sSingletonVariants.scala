package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

import scala.collection.{Set, Map}

object sSingletonVariants {
  def apply(vds: VariantDataset): Set[Variant] = {
    nNonRefPerVariant(vds)
      .filter(_._2 == 1)
      .keys
      .collect().toSet
  }
}

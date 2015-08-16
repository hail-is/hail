package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

object nNonRefPerVariant extends VariantMethod[Int] {
  def name = "nNonRef"

  def apply(vds: VariantDataset): Map[Variant, Int] = {
    nGenotypeVectorPerVariant(vds)
      .mapValues(a => a(1) + a(2))
  }
}

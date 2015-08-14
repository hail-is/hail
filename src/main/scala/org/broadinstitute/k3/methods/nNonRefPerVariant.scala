package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

import scala.collection.Map

object nNonRefPerVariant {
  def apply(vds: VariantDataset): Map[Variant, Int] = {
    nGenotypeVectorPerVariant(vds)
      .mapValues(a => a(1) + a(2))
  }
}

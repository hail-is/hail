package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

// FIXME: can we pattern match the vector?
object rHeterozygosityPerVariant extends VariantMethod[Double] {
  def name = "rHeterozygosity"

  def apply(vds: VariantDataset): Map[Variant, Double] = {
    nGenotypeVectorPerVariant(vds).mapValues(a => {
      val nCalled = a(0) + a(1) + a(2)
      if (nCalled != 0) a(1).toDouble / nCalled else -1
    })
  }
}

package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

// FIXME: need to account for all HomRef
object nNonRefPerSample extends SampleMethod[Int] {
  def name = "nNonRef"

  def apply(vds: VariantDataset): Map[Int, Int] = {
    nGenotypeVectorPerSample(vds)
      .mapValues(a => a(1) + a(2))
  }
}

package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

import scala.collection.Map

// FIXME: need to account for all HomRef
object nNonRefPerSample {
  def apply(vds: VariantDataset): Map[Int, Int] = {
    nGenotypeVectorPerSample(vds)
      .mapValues(a => a(1) + a(2))
  }
}

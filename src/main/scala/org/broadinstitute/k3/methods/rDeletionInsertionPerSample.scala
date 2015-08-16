package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

// FIXME: can we pattern match (snp, ins, del) against vector?
object rDeletionInsertionPerSample extends SampleMethod[Double] {
  def name = "rDeletionInsertion"

  def apply(vds: VariantDataset): Map[Int, Double] = {
    nVariantTypeVectorPerSample(vds)
      .mapValues( a => if (a(1) != 0) a(2).toDouble / a(1) else -1)
  }
}

package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

// FIXME: can we pattern match (snp, ins, del) against vector?
object rDeletionInsertionPerSample extends SampleMethod[Double] {
  def name = "rDeletionInsertion"

  def apply(vds: VariantDataset): Map[Int, Double] = {
    nVariantTypeVectorPerSample(vds)
      .mapValues( a => if (a._2 != 0) a._3.toDouble / a._2 else -1)
  }
}

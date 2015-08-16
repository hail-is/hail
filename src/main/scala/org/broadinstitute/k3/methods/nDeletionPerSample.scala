package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

object nDeletionPerSample extends SampleMethod[Int] {
  def name = "nDeletion"

  def apply(vds: VariantDataset): Map[Int, Int] = {
    vds
      .mapValuesWithKeys((v, s, g) => if (g.isNonRef && v.isDeletion) 1 else 0)
      .foldBySample(0)(_ + _)
  }
}

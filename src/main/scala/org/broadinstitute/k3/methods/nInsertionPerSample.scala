package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

object nInsertionPerSample extends SampleMethod[Int] {
  def name = "nInsertion"

  def apply(vds: VariantDataset): Map[Int, Int] = {
    vds
      .mapValuesWithKeys((v, s, g) => if (g.isNonRef && v.isInsertion) 1 else 0)
      .foldBySample(0)(_ + _)
  }
}




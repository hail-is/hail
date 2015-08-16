package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

object nSNPPerSample extends SampleMethod[Int] {
  def name = "nSNP"

  def apply(vds: VariantDataset): Map[Int, Int] = {
    vds
      .mapValuesWithKeys((v, s, g) => if (g.isNonRef && v.isSNP) 1 else 0)
      .foldBySample(0)(_ + _)
  }
}
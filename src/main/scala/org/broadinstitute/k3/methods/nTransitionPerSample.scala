package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

object nTransitionPerSample extends SampleMethod[Int] {
  def name = "nTransition"

  def apply(vds: VariantDataset): Map[Int, Int] = {
    vds
      .mapValuesWithKeys((v, s, g) => if (g.isNonRef && v.isTransition) 1 else 0)
      .foldBySample(0)(_ + _)
  }
}

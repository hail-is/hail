package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

object nHomVarPerSample extends SampleMethod[Int] {
  def name = "nHomVar"

  def apply(vds: VariantDataset): Map[Int, Int] = {
    vds
      .mapValues(g => if (g.isHomVar) 1 else 0)
      .foldBySample(0)(_ + _)
  }
}

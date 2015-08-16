package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

object nHetPerSample extends SampleMethod[Int] {
  def name = "nHet"

  def apply(vds: VariantDataset): Map[Int, Int] = {
    vds
      .mapValues(g => if (g.isHet) 1 else 0)
      .foldBySample(0)(_ + _)
  }
}

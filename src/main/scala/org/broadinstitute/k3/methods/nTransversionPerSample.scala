package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

object nTransversionPerSample extends SampleMethod[Int] { // should compute with transitions
  def name = "nTransversion"

  def apply(vds: VariantDataset): Map[Int, Int] = {
    vds
      .mapValuesWithKeys((v, s, g) => if (g.isNonRef && v.isTransversion) 1 else 0)
      .foldBySample(0)(_ + _)
  }
}




package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

import scala.collection.Map

object nSamplePerVariant {
  def apply(vds: VariantDataset): Map[Variant, Int] = {
    vds
      .mapValues(g => 1)
      .foldByVariant(0)(_ + _)
  }
}

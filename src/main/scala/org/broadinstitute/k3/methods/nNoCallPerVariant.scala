package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

import scala.collection.Map

object nNoCallPerVariant {
  def apply(vds: VariantDataset): Map[Variant, Int] = {
    vds
      .mapValues(g => if (g.notCalled) 1 else 0)
      .foldByVariant(0)(_ + _)
  }
}

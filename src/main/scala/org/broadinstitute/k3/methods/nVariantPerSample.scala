package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

import scala.collection.Map

object nVariantPerSample {
  def apply(vds: VariantDataset): Map[Int, Int] = {
    vds
      .mapValues(g => 1)
      .foldBySample(0)(_ + _)
  }
}

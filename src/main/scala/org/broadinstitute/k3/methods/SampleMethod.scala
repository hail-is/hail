package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

abstract class SampleMethod[+T] {
  def name: String
  def apply(vds: VariantDataset): Map[Int, T]
}

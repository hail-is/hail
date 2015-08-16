package org.broadinstitute.k3.methods

import org.apache.spark.rdd.RDD
import org.broadinstitute.k3.variant._

// FIXME
abstract class VariantMethod[T] {
  def name: String
  def apply(vds: VariantDataset): RDD[(Variant, T)]
}

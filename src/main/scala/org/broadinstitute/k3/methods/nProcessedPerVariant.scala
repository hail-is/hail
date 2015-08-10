package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

import scala.collection.Map

// FIXME: this breaks the abstraction barrier
object nProcessedPerVariant {
  def apply(vds: VariantDataset): Long = {
    vds.count()
  }
}

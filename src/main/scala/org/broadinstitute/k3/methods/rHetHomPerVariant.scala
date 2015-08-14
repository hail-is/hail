package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

import scala.collection.Map

// FIXME: can we pattern match (hr, het, hv, nc) against vector?
object rHetHomPerVariant {
  def apply(vds: VariantDataset): Map[Variant, Double] = {
    nGenotypeVectorPerVariant(vds).mapValues(a => {
      val nHom = a(0) + a(2)
      if (nHom != 0) a(1).toDouble / nHom else -1
    })
  }
}

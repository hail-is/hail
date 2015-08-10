package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

import scala.collection.Map

// FIXME: need to account for all HomRef, requires explicit toString for print via tuple in main
object rHetHomPerSample {
  def apply(vds: VariantDataset): Map[Int, Double] = {
    nGenotypePerSampleVector(vds).mapValues(a => {
      val nHom = a(0) + a(2)
      if (nHom != 0) a(1).toDouble / nHom else -1
    })
  }
}

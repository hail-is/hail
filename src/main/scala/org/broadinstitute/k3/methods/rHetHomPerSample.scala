package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

// FIXME: need to account for all HomRef, can we pattern match (hr, het, hv, nc) against vector?
object rHetHomPerSample extends SampleMethod[Double] {
  def name = "rHetHom"

  def apply(vds: VariantDataset): Map[Int, Double] = {
    nGenotypeVectorPerSample(vds).mapValues(a => {
      val nHom = a._1 + a._3
      if (nHom != 0) a._2.toDouble / nHom else -1
    })
  }
}

package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

import scala.collection.Map

object rTiTvPerSample {
  def apply(vds: VariantDataset): Map[Int, Double] = {
    nTiTvPerSample(vds)
      .mapValues { case (nTi, nTv) => if (nTv != 0) nTi.toDouble / nTv else -1 }
  }
}




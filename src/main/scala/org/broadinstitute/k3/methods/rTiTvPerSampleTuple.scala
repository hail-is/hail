package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

import scala.collection.Map

object rTiTvPerSampleTuple { // not optimized, can do one pass and use that tv is complementary to ti among SNPs
  def apply(vds: VariantDataset): Map[Int, Double] = {
    val nTiTv = nTiTvPerSampleTuple(vds)

    for {
      (s, (nTi, nTv)) <- nTiTv
    } yield s -> (if (nTv != 0) nTi.toDouble / nTv else -1)
  }
}




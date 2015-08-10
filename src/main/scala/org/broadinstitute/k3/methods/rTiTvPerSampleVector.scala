package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

import scala.collection.Map

object rTiTvPerSampleVector { // not optimized, can do one pass and use that tv is complementary to ti among SNPs
  def apply(vds: VariantDataset): Map[Int, Double] = {
    val nTiTv = nTiTvPerSampleVector(vds)

    for {
      (s, tiTv) <- nTiTv
    } yield s -> (if (tiTv(1) != 0) tiTv(0).toDouble / tiTv(1) else -1)
  }
}

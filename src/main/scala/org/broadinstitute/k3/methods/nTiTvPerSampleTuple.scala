package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

import scala.collection.Map

object nTiTvPerSampleTuple {
  def apply(vds: VariantDataset): Map[Int, (Int, Int)] = {
    vds
      .aggregateBySampleWithKeys((0,0))({ case ((nTi, nTv), v, s, g) =>
      if ((g.isHet || g.isHomVar) && v.isSNP) {
        if (v.isTransition) // non-optimal: checking isSNP twice
          (nTi + 1, nTv)
        else
          (nTi, nTv + 1)
      } else
        (nTi, nTv)},
    (x, y) => (x._1 + y._1, x._2 + y._2) )
  }
}




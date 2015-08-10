package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

import scala.collection.Map

object nTiTvPerSampleVector {
  def apply(vds: VariantDataset): Map[Int, Vector[Int]] = {
    vds
      .aggregateBySampleWithKeys(Vector(0,0))({ case (a, v, s, g) =>
      if ((g.isHet || g.isHomVar) && v.isSNP) {
        if (v.isTransition) // non-optimal: checking isSNP twice
          a.updated(0, a(0) + 1)
        else
          a.updated(1, a(1) + 1)
      } else
        a},
    (x, y) => Vector(x(0) + y(0), x(1) + y(1)))
  }
}




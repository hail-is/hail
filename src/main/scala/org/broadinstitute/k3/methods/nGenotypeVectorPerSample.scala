package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

// FIXME: need to account for all HomRef
object nGenotypeVectorPerSample extends SampleMethod[(Int, Int, Int, Int)] {
  def name = "nGenotypeVector"

  def apply(vds: VariantDataset): Map[Int, (Int, Int, Int, Int)] = {
    vds
      .mapValues(g =>
      if (g.isHomRef)
        (1,0,0,0)
      else if (g.isHet)
        (0,1,0,0)
      else if (g.isHomVar)
        (0,0,1,0)
      else {
        assert(g.isNotCalled)
        (0,0,0,1)
      })
      .foldBySample((0,0,0,0))((x, y) => (x._1 + y._1, x._2 + y._2, x._3 + y._3, x._4 + y._4))
  }
}



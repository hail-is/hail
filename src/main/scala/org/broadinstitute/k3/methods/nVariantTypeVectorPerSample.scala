package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

object nVariantTypeVectorPerSample extends SampleMethod[(Int, Int, Int)] {
  def name = "nVariantTypeVector"

  def apply(vds: VariantDataset): Map[Int, (Int, Int, Int)] = {
    vds
      .mapValuesWithKeys((v,s,g) =>
      if (g.isNonRef) {
        if (v.isSNP)
          (1, 0, 0)
        else if (v.isInsertion)
          (0, 1, 0)
        else if (v.isDeletion)
          (0, 0, 1)
        else {
          assert(v.isComplex) // until complex is defined, will throw error
          (0, 0, 0)
        }
      }
      else
        (0,0,0))
      .foldBySample((0,0,0))((x, y) => (x._1 + y._1, x._2 + y._2, x._3 + y._3))
  }
}

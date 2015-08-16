package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

object nVariantTypeVectorPerSample extends SampleMethod[Vector[Int]]{
  def name = "nVariantTypeVector"

  def apply(vds: VariantDataset): Map[Int, Vector[Int]] = {
    vds
      .mapValuesWithKeys((v,s,g) =>
      if (g.isNonRef) {
        if (v.isSNP)
          Vector(1, 0, 0)
        else if (v.isInsertion)
          Vector(0, 1, 0)
        else if (v.isDeletion)
          Vector(0, 0, 1)
        else {
          assert(v.isComplex) // until complex is defined, will throw error
          Vector(0, 0, 0)
        }
      }
      else
        Vector(0,0,0))
      .foldBySample(Vector(0,0,0))((x, y) => Vector(x(0) + y(0), x(1) + y(1), x(2) + y(2)))
  }
}

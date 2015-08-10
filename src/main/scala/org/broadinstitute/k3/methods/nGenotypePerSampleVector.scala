package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

import scala.collection.Map

object nGenotypePerSampleVector {
  def apply(vds: VariantDataset): Map[Int, Vector[Int]] = {
    vds
      .aggregateBySample(Vector(0,0,0,0))((a, g) =>
      if (g.isHomRef)
        a.updated(0, a(0) + 1)
      else if (g.isHet)
        a.updated(1, a(1) + 1)
      else if (g.isHomVar)
        a.updated(2, a(2) + 1)
      else // noCall
        a.updated(3, a(3) + 1),
        (x, y) => Vector(x(0) + y(0), x(1) + y(1), x(2) + y(2), x(3) + y(3)))
  }
}



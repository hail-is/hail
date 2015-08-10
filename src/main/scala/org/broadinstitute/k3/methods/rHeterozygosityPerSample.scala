package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

import scala.collection.Map

// FIXME: seems to require explicit toString for printing via tuple in main
object rHeterozygosityPerSample {
  def apply(vds: VariantDataset): Map[Int, Double] = {
    nGenotypeVectorPerSample(vds).mapValues(a => {
      val nCalled = a(0) + a(1) + a(2)
      if (nCalled != 0) a(1).toDouble / nCalled else -1
    })
  }
}

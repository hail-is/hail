package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

// FIXME: seems to require explicit toString for printing via tuple in main
object rHeterozygosityPerSample extends SampleMethod[Double]{
  def name = "rHeterozygosity"

  def apply(vds: VariantDataset): Map[Int, Double] = {
    nGenotypeVectorPerSample(vds).mapValues(a => {
      val nCalled = a(0) + a(1) + a(2)
      if (nCalled != 0) a(1).toDouble / nCalled else -1
    })
  }
}

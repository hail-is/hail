package org.broadinstitute.k3.methods

import org.apache.spark.rdd.RDD
import org.broadinstitute.k3.variant._

// FIXME: can we pattern match the vector?
object rHeterozygosityPerVariant extends VariantMethod[Double] {
  def name = "rHeterozygosity"

  def apply(vds: VariantDataset): RDD[(Variant, Double)] = {
    nGenotypeVectorPerVariant(vds).mapValues(a => {
      val nCalled = a._1 + a._2 + a._3
      if (nCalled != 0) a._2.toDouble / nCalled else -1
    })
  }
}

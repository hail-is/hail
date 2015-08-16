package org.broadinstitute.k3.methods

import org.apache.spark.rdd.RDD
import org.broadinstitute.k3.variant._

// FIXME: can we pattern match (hr, het, hv, nc) against vector?
object rHetHomPerVariant extends VariantMethod[Double] {
  def name = "rHetHom"

  def apply(vds: VariantDataset): RDD[(Variant, Double)] = {
    nGenotypeVectorPerVariant(vds).mapValues(a => {
      val nHom = a._1 + a._3
      if (nHom != 0) a._2.toDouble / nHom else -1
    })
  }
}

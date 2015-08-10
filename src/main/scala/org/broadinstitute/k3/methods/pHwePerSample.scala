package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

import scala.collection.Map

object pHwePerSample {  // FIXME currently just compute chi2 stat, and not including all HomRef sites
  def apply(vds: VariantDataset): Map[Int, Double] = {
    nGenotypePerSampleVector(vds)
      .mapValues(a => chiSq(Array(a(0), a(1), a(2))))
  }

  def chiSq(observed: Array[Int]): Double = { // FIXME handle div by 0
    val total = observed.sum.toDouble
    val p = (observed(1) + 2 * observed(2)) / total
    val q = 1 - p
    val expected = Array[Double](q * q * total, 2 * p * q * total, p * p * total)

    (0 to 2).map(i => math.pow(observed(i) - expected(i), 2) / expected(i)).sum
  }
}

package org.broadinstitute.k3.methods

import org.apache.spark.rdd.RDD
import org.broadinstitute.k3.variant._

import org.broadinstitute.k3.Utils._

// FIXME currently just compute chi2 stat, need tests
object pHwePerVariant extends VariantMethod[Double] {
  def name = "nHWE"

  def apply(vds: VariantDataset): RDD[(Variant, Double)] = {
    nGenotypeVectorPerVariant(vds)
      .mapValues(a => chiSq(Vector(a._1, a._2, a._3)))
  }

  def chiSq(observed: Vector[Int]): Double = { // FIXME handle div by 0
    require(observed.length == 3)
    val total = observed.sum.toDouble
    val p = (observed(1) + 2 * observed(2)) / total
    val q = 1 - p
    val expected = Vector(q * q * total, 2 * p * q * total, p * p * total)

    // FIXME: why is this not getting inferred?
    observed.zipWith[Double, Double](expected, (o, e) => (o - e) * (o - e) / e).sum
  }
}

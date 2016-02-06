package org.broadinstitute.hail.methods

import breeze.linalg._
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.variant._

case class DenseStats(x: DenseVector[Double], xx: Double, xy: Double, nMissing: Int)

object DenseStats {
  def name = "SparseStats"

  def apply(hcs: HardCallSet, ped: Pedigree, cov: CovariateData): RDD[(Variant, DenseStats)] = {
    // require(ped.trios.forall(_.pheno.isDefined))
    val sampleCovRow = cov.covRowSample.zipWithIndex.toMap

    val n = cov.data.rows
    val k = cov.data.cols
    val d = n - k - 2
    if (d < 1)
      throw new IllegalArgumentException(n + " samples and " + k + " covariates implies " + d + " degrees of freedom.")

    val sc = hcs.rdd.sparkContext
    val sampleCovRowBc = sc.broadcast(sampleCovRow)
    val samplesWithCovDataBc = sc.broadcast(sampleCovRow.keySet)

    val samplePheno = ped.samplePheno
    val yArray = (0 until n).flatMap(cr => samplePheno(cov.covRowSample(cr)).map(_.toString.toDouble)).toArray
    val y = DenseVector[Double](yArray)
    val yBc = sc.broadcast(y)

    hcs.rdd.mapValues(_.denseStats(yBc.value, n))
  }
}
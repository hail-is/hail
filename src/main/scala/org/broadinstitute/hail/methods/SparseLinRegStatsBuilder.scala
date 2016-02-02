package org.broadinstitute.hail.methods

import breeze.linalg._
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.variant._
import scala.collection.mutable.ArrayBuffer

class SparseLinRegStatsBuilder extends Serializable {
  private val rowsX = ArrayBuffer[Int]()
  private val valsX = ArrayBuffer[Double]()
  private var sumX = 0
  private var sumXX = 0
  private var sumXY = 0.0
  private val missingRows = ArrayBuffer[Int]()

  def merge(row: Int, gt: Int, y: DenseVector[Double]): SparseLinRegStatsBuilder = {
    gt match {
      case 0 =>
      case 1 =>
        rowsX += row
        valsX += 1.0
        sumX += 1
        sumXX += 1
        sumXY += y(row)
      case 2 =>
        rowsX += row
        valsX += 2.0
        sumX += 2
        sumXX += 4
        sumXY += 2 * y(row)
      case 3 =>
        missingRows += row
      case _ => throw new IllegalArgumentException(s"Genotype value $gt must be 0, 1, 2, 3.")
    }
    this
  }

  def merge(that:  SparseLinRegStatsBuilder):  SparseLinRegStatsBuilder = {
    rowsX ++= that.rowsX.toArray
    valsX ++= that.valsX.toArray
    sumX += that.sumX
    sumXX += that.sumXX
    sumXY += that.sumXY
    missingRows ++= that.missingRows.toArray

    this
  }

  def stats(y: DenseVector[Double], n: Int): SparseStats = {
    val missingRowsArray = missingRows.toArray
    val nMissing = missingRowsArray.size
    val nPresent = n - nMissing

    val meanX = sumX.toDouble / nPresent
    rowsX ++= missingRowsArray
    (0 until nMissing).foreach(_ => valsX += meanX)

    val x = new SparseVector[Double](rowsX.toArray, valsX.toArray, n)
    val xx = sumXX + meanX * meanX * nMissing
    val xy = sumXY + meanX * missingRowsArray.iterator.map(y(_)).sum

    SparseStats(x, xx, xy, nMissing)
  }
}

case class SparseStats(x: DenseVector[Double], xx: Double, xy: Double, nMissing: Int)

object SparseStats {
  def name = "SparseStats"

  def apply(hcs: HardCallSet, ped: Pedigree, cov: CovariateData): RDD[(Variant, SparseStats)] = {
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

    hcs.rdd.mapValues(_.linRegStats(yBc.value, n))

    //.filterSamples { case (s, sa) => samplesWithCovDataBc.value.contains(s) }
  }
}
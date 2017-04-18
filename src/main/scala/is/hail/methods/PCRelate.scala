package is.hail.methods

import org.apache.spark.rdd.RDD
import is.hail.utils._
import is.hail.keytable.KeyTable
import is.hail.variant.{Variant, VariantDataset}
import is.hail.distributedmatrix.DistributedMatrix
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed._

import scala.collection.generic.CanBuildFrom
import scala.language.higherKinds
import scala.language.implicitConversions

object PCRelate {
  case class Result[M: DistributedMatrix](phiHat: M)

  def apply[M: DistributedMatrix](vds: VariantDataset, pcs: DenseMatrix): Result[M] = {
    println(s"Starting pc relate ${System.nanoTime()}")
    val g = vdsToMeanImputedMatrix(vds).cache()

    println(s"mean imputed ${System.nanoTime()}")

    val beta = fitBeta(g, pcs)

    println(s"fitted Beta ${System.nanoTime()}")
    val phihat = phiHat(DistributedMatrix[M].from(g), muHat(pcs, beta))

    println(s"calculated phiHat ${System.nanoTime()}")
    Result(phihat)
  }

  def vdsToMeanImputedMatrix[M: DistributedMatrix](vds: VariantDataset): RDD[Array[Double]] = {
    val rdd = vds.rdd.mapPartitions { part =>
      part.map { case (v, (va, gs)) =>
        val goptions = gs.map(_.gt.map(_.toDouble)).toArray
        val defined = goptions.flatMap(x => x)
        val mean = defined.sum / defined.length
        goptions.map(_.getOrElse(mean))
      }
    }
    rdd
  }

  /**
    *  g: SNP x Sample
    *  pcs: Sample x D
    *
    *  result: (SNP x (D+1))
    */
  def fitBeta[M: DistributedMatrix](g: RDD[Array[Double]], pcs: DenseMatrix): M = {
    val aa = g.sparkContext.broadcast(pcs.rowIter.map(_.toArray).toArray)
    val rdd = g.map { row =>
      val ols = new OLSMultipleLinearRegression()
      ols.newSampleData(row, aa.value)
      ols.estimateRegressionParameters()
    }
    DistributedMatrix[M].from(rdd)
  }

  private def clipToInterval(x: Double): Double =
    if (x <= 0)
      Double.MinPositiveValue
    else if (x >= 1)
      1 - Double.MinPositiveValue
    else
      x

  /**
    *  pcs: Sample x D
    *  betas: SNP x (D+1)
    *  beta0: SNP
    *
    *  result: SNP x Sample
    */
  def muHat[M: DistributedMatrix](pcs: DenseMatrix, beta: M): M = {
    val dm = DistributedMatrix[M]
    import dm.ops._

    val pcsArray = pcs.toArray
    val pcsWithInterceptArray = new Array[Double](pcs.numRows * (pcs.numCols + 1))
    var i = 0
    while (i < pcs.numRows) {
      pcsWithInterceptArray(i) = 1
      i += 1
    }

    i = 0
    while (i < pcs.numRows * pcs.numCols) {
      pcsWithInterceptArray(pcs.numRows + i) = pcsArray(i)
      i += 1
    }

    val pcsWithIntercept =
      new DenseMatrix(pcs.numRows, pcs.numCols + 1, pcsWithInterceptArray)

    dm.map(clipToInterval _)((beta * pcsWithIntercept.transpose) / 2.0)
  }

  /**
    * g: SNP x Sample
    * muHat: SNP x Sample
    **/
  def phiHat[M: DistributedMatrix](g: M, muHat: M): M = {
    val dm = DistributedMatrix[M]
    import dm.ops._

    val gMinusMu = g :-: (muHat * 2.0)
    val varianceHat = muHat :*: (1.0 - muHat)

    // println("means")
    // println(dm.toBlockRdd(muHat * 2.0).take(1)(0))
    // println("numerator")
    // println(dm.toBlockRdd(gMinusMu.t * gMinusMu).take(1)(0))
    // println("denominator")
    // println(dm.toBlockRdd((varianceHat.t * varianceHat).sqrt).take(1)(0))

    ((gMinusMu.t * gMinusMu) :/: (varianceHat.t * varianceHat).sqrt) / 4.0
  }

  def f[M: DistributedMatrix](phiHat: M): Array[Double] =
    DistributedMatrix[M].diagonal(phiHat).map(2.0 * _ - 1.0)

  def k2[M: DistributedMatrix](f: Array[Double], gD: M, mu: M): M = {
    val dm = DistributedMatrix[M]
    import dm.ops._

    val normalizedGD = gD :-: ((mu :*: (1.0 - mu)) --* (f.map(1 + _)))
    val variance = mu * (1.0 - mu)

    (normalizedGD.t * normalizedGD) :/: (variance.t * variance)
  }

  def k0[M: DistributedMatrix](f: Array[Double], gD: M, mu: M): M =
    ???

  def k1[M: DistributedMatrix](k2: M, k0: M): M = {
    val dm = DistributedMatrix[M]
    import dm.ops._

    1.0 - (k2 :-: k0)
  }
}

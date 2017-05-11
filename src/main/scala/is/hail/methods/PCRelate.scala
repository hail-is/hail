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
  case class Result[M: DistributedMatrix](phiHat: M, k0: M, k1: M, k2: M)

  def apply[M: DistributedMatrix](vds: VariantDataset, pcs: DenseMatrix): Result[M] = {
    val g = vdsToMeanImputedMatrix(vds).cache()

    val beta = fitBeta(g, pcs)

    val mu = muHat(pcs, beta)

    val phihat = phiHat(DistributedMatrix[M].from(g), mu)

    val kTwo = k2(f(phihat), DistributedMatrix[M].from(dominance(g)), mu)

    val kZero = k0(phihat, mu, kTwo, ibs0(vds))

    Result(phihat, kZero, k1(kTwo, kZero), kTwo)
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
    *  beta: SNP x (D+1)
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

    val gMinusMu = g :- (muHat * 2.0)
    val varianceHat = muHat :* (1.0 - muHat)

    ((gMinusMu.t * gMinusMu) :/ (varianceHat.t.sqrt * varianceHat.sqrt)) / 4.0
  }

  def f[M: DistributedMatrix](phiHat: M): Array[Double] =
    DistributedMatrix[M].diagonal(phiHat).map(2.0 * _ - 1.0)

  def k2[M: DistributedMatrix](f: Array[Double], gD: M, mu: M): M = {
    val dm = DistributedMatrix[M]
    import dm.ops._

    val normalizedGD = gD :- ((mu :* (1.0 - mu)) --* (f.map(1 + _)))
    val variance = mu :* (1.0 - mu)

    (normalizedGD.t * normalizedGD) :/ (variance.t * variance)
  }

  /**
    * muHat: SNP x Sample
    *
    **/
  private def sqr(x: Double): Double = x * x
  private val cutoff = math.pow(2.0, (-5.0/2.0))
  private def _k0(phiHat: Double, denom: Double, k2: Double, ibs0: Double) =
    if (phiHat <= cutoff)
      1 - 4 * phiHat + k2
    else
      ibs0 / denom
  def k0[M: DistributedMatrix](phiHat: M, mu: M, k2: M, ibs0: M): M = {
    val dm = DistributedMatrix[M]
    import dm.ops._

    val mu2 = dm.map(sqr _)(mu)
    val oneMinusMu2 = dm.map(sqr _)(1.0 - mu)

    val denom = (mu2.t * oneMinusMu2) :+ (oneMinusMu2.t * mu2)

    dm.map4(_k0)(phiHat, denom, k2, ibs0)
  }

  def k1[M: DistributedMatrix](k2: M, k0: M): M = {
    val dm = DistributedMatrix[M]
    import dm.ops._

    1.0 - (k2 :+ k0)
  }

  def ibs0[M: DistributedMatrix](vds: VariantDataset): M =
    DistributedMatrix[M].from(IBD.ibs(vds)._1)

  // FIXME: fuse with vdsToMeanImputedMatrix somehow, no need to recalculate means
  def dominance(g: RDD[Array[Double]]): RDD[Array[Double]] = g.map { a =>
    val p = (a.sum / a.size)
    a.map(_ - p)
  }
}

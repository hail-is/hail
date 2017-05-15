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

  def maybefast[M: DistributedMatrix](vds: VariantDataset, pcs: DenseMatrix): Result[M] = {
    val dm = DistributedMatrix[M]
    import dm.ops._

    val g = vdsToMeanImputedMatrix(vds).cache()

    val beta = fitBeta(g, pcs)

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

    val mu = dm.map(clipToInterval _)((beta * pcsWithIntercept.transpose) / 2.0).cache()

    val blockedG = DistributedMatrix[M].from(g).cache()
    val gMinusMu = (blockedG :- (mu * 2.0)).cache()
    val variance = (mu :* (1.0 - mu)).cache()

    val phi = (((gMinusMu.t * gMinusMu) :/ (variance.t.sqrt * variance.sqrt)) / 4.0).cache()

    val gD = DistributedMatrix[M].map2({
      case (g, mu) =>
        if (g == 0.0) mu
        else if (g == 1.0) 0.0
        else if (g == 2.0) 1.0 - mu
        else g
    })(blockedG, mu)

    val normalizedGD = (gD :- (variance --* (f(phi).map(1 + _)))).cache()

    val kTwo = ((normalizedGD.t * normalizedGD) :/ (variance.t * variance)).cache()

    val mu2 = dm.map(sqr _)(mu).cache()
    val oneMinusMu2 = dm.map(sqr _)(1.0 - mu).cache()

    val denom = (mu2.t * oneMinusMu2) :+ (oneMinusMu2.t * mu2)

    def _k0(phiHat: Double, denom: Double, k2: Double, ibs0: Double) =
      if (phiHat <= cutoff)
        1 - 4 * phiHat + k2
      else
        ibs0 / denom
    val kZero = dm.map4(_k0)(phi, denom, kTwo, ibs0(vds)).cache()

    Result(phi, kZero, 1.0 - (kTwo :+ kZero), kTwo)
  }

  def apply[M: DistributedMatrix](vds: VariantDataset, pcs: DenseMatrix): Result[M] = {
    val g = vdsToMeanImputedMatrix(vds).cache()

    val beta = fitBeta(g, pcs)

    val mu = muHat(pcs, beta)

    val blockedG = DistributedMatrix[M].from(g)
    val phihat = phiHat(blockedG, mu)

    val kTwo = k2(f(phihat), DistributedMatrix[M].map2({ case (g, mu) => if (g == 0.0) mu else if (g == 1.0) 0.0 else if (g == 2.0) 1.0 - mu else g })(blockedG, mu), mu)

    val kZero = k0(phihat, mu, kTwo, ibs0(vds))

    // println(DistributedMatrix[M].toLocalMatrix(kTwo))

    Result(phihat, kZero, k1(kTwo, kZero), kTwo)
  }

  def vdsToMeanImputedMatrix[M: DistributedMatrix](vds: VariantDataset): RDD[Array[Double]] = {
    val nSamples = vds.nSamples
    val rdd = vds.rdd.mapPartitions { part =>
      part.map { case (v, (va, gs)) =>
        var sum = 0
        var nNonMissing = 0
        val missingIndices = new ArrayBuilder[Int]()
        val a = new Array[Double](nSamples)

        var i = 0
        val it = gs.hardCallIterator
        while (it.hasNext) {
          val gt = it.next()
          if (gt == -1) {
            missingIndices += i
          } else {
            sum += gt
            a(i) = gt
            nNonMissing += 1
          }
          i += 1
        }

        val mean = sum.toDouble / nNonMissing

        for (i <- missingIndices.result()) {
          a(i) = mean
        }

        a
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
    val rdd = g.map { a =>
      val ols = new OLSMultipleLinearRegression()
      ols.newSampleData(a, aa.value)
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

    // println(pcsWithIntercept)
    // println(dm.toLocalMatrix(beta))

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

    // println("mu")
    // println(dm.toLocalMatrix(mu))
    // println("gD")
    // println(dm.toLocalMatrix(gD))
    // println("f.map(1 + _)")
    // println(f.map(1 + _): IndexedSeq[Double])
    // println("mu :* (1.0 - mu)")
    // println(dm.toLocalMatrix(mu :* (1.0 - mu)))
    // println("(mu :* (1.0 - mu)) --* f.map(1 + _)")
    // println(dm.toLocalMatrix((mu :* (1.0 - mu)) --* (f.map(1 + _))))

    val normalizedGD = gD :- ((mu :* (1.0 - mu)) --* (f.map(1 + _)))
    val variance = mu :* (1.0 - mu)

    // println("numer")
    // println(dm.toLocalMatrix(normalizedGD.t * normalizedGD))
    // println("denom")
    // println(dm.toLocalMatrix(variance.t * variance))

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

}

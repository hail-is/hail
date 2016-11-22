package org.broadinstitute.hail.methods

import breeze.linalg._
import org.apache.commons.math3.distribution.TDistribution
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.variant._

object LinearRegressionHcs {
  def apply(
    hcs: HardCallSet,
    y: DenseVector[Double],
    cov: Option[DenseMatrix[Double]],
    sampleFilter: Array[Boolean],
    reduceSampleIndex: Array[Int],
    minMAC: Int = 0,
    maxMAC: Int = Int.MaxValue): LinearRegressionHcs = {

    val n = hcs.nSamples
    val n0 = y.size
    val k = if (cov.isDefined) cov.get.cols else 0
    val d = n0 - k - 2

    assert(n >= n0)
    assert(cov.forall(_.rows == n0))

    if (d < 1)
      fatal(s"$n0 samples and $k ${plural(k, "covariate")} with intercept implies $d degrees of freedom.")

    info(s"Running linreg on $n0 samples with $k sample ${plural(k, "covariate")}...")

    val covAndOnes: DenseMatrix[Double] = cov match {
      case Some(dm) => DenseMatrix.horzcat(dm, DenseMatrix.ones[Double](n0, 1))
      case None => DenseMatrix.ones[Double](n0, 1)
    }

    val qt = qr.reduced.justQ(covAndOnes).t
    val qty = qt * y

    val sc = hcs.sparkContext
    val yBc = sc.broadcast(y)
    val qtBc = sc.broadcast(qt)
    val qtyBc = sc.broadcast(qty)
    val yypBc = sc.broadcast((y dot y) - (qty dot qty))
    val tDistBc = sc.broadcast(new TDistribution(null, d.toDouble))

    val sampleFilterBc = sc.broadcast(sampleFilter)
    val reduceSampleIndexBc = sc.broadcast(reduceSampleIndex)

    new LinearRegressionHcs(hcs
      .rdd
      .mapValues { cs => // FIXME: only three are necessary
        val GtVectorAndStats(x, nHomRef, nHet, nHomVar, nMissing, meanX) = cs.hardStats(n, n0, sampleFilterBc.value, reduceSampleIndexBc.value)

        val nPresent = n0 - nMissing
        val mac = nHet + nHomVar * 2

        if (nHomRef == nPresent || nHet == nPresent || nHomVar == nPresent || mac < minMAC || mac > maxMAC)
          None
        else {
          val xx = nHet + nHomVar * 4 + nMissing * meanX * meanX
          val qtx = qtBc.value * x
          val qty = qtyBc.value
          val xxp: Double = xx - (qtx dot qtx)
          val xyp: Double = (x dot yBc.value) - (qtx dot qty)
          val yyp: Double = yypBc.value

          if (xxp == 0d)
            None
          else {
            val b: Double = xyp / xxp
            val se = math.sqrt((yyp / xxp - b * b) / d)
            if (se == 0)
              None
            else {
              val t = b / se
              val p = 2 * tDistBc.value.cumulativeProbability(-math.abs(t))
              if (p.isNaN)
                None
              else
                Some(LinRegStatsHcs(b, se, t, p))
            }
          }
        }
      }
    )
  }
}

case class LinearRegressionHcs(rdd: RDD[(Variant, Option[LinRegStatsHcs])])

object LinRegStatsHcs {
  def `type`: Type = TStruct(
    ("beta", TDouble),
    ("se", TDouble),
    ("tstat", TDouble),
    ("pval", TDouble))
}

case class LinRegStatsHcs(beta: Double, se: Double, t: Double, p: Double) {
  def toAnnotation: Annotation = Annotation(beta, se, t, p)
}

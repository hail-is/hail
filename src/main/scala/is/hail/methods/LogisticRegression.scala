package is.hail.methods

import breeze.linalg._
import org.apache.spark.rdd.RDD
import is.hail.utils._
import is.hail.annotations.Annotation
import is.hail.stats._
import is.hail.variant._
import is.hail.expr._

object LogisticRegression {

  def apply(vds: VariantDataset, pathVA: List[String], completeSamples: IndexedSeq[String], y: DenseVector[Double], cov: DenseMatrix[Double], logRegTest: LogisticRegressionTest): VariantDataset = {
    require(cov.rows == y.size)
    require(completeSamples.size == y.size)
    require(y.forall(yi => yi == 0 || yi == 1))
    require {val sumY = sum(y); sumY > 0 && sumY < y.size}

    val n = y.size
    val k = cov.cols
    val d = n - k - 1

    if (d < 1)
      fatal(s"$n samples and $k ${plural(k, "covariate")} including intercept implies $d degrees of freedom.")

    info(s"Running logreg on $n samples with $k ${plural(k, "covariate")} including intercept...")

    val nullModel = new LogisticRegressionModel(cov, y)
    val nullFit = nullModel.nullFit(nullModel.bInterceptOnly())

    if (!nullFit.converged)
      fatal("Failed to fit logistic regression null model (covariates only): " + (
        if (nullFit.exploded)
         s"exploded at Newton iteration ${nullFit.nIter}"
        else
         "Newton iteration failed to converge"))

    val completeSamplesSet = completeSamples.toSet
    assert(completeSamplesSet.size == completeSamples.size)
    val sampleMask = vds.sampleIds.map(completeSamplesSet).toArray

    val sc = vds.sparkContext
    val sampleMaskBc = sc.broadcast(sampleMask)
    val yBc = sc.broadcast(y)
    val covBc = sc.broadcast(cov)
    val nullFitBc = sc.broadcast(nullFit)
    val logRegTestBc = sc.broadcast(logRegTest) // Is this worth broadcasting?

    val (newVAS, inserter) = vds.insertVA(logRegTest.`type`, pathVA)

    vds.mapAnnotations{ case (v, va, gs) =>
      val maskedGts = gs.zipWithIndex.filter{ case (g, i) => sampleMaskBc.value(i) }.map(_._1.gt)

      val logRegAnnot =
        buildGtColumn(maskedGts).map { gts =>
          val X = DenseMatrix.horzcat(gts, covBc.value)
          logRegTestBc.value.test(X, yBc.value, nullFitBc.value).toAnnotation
        }

      inserter(va, logRegAnnot)
    }.copy(vaSignature = newVAS)
  }

  def buildGtColumn(gts: Iterable[Option[Int]]): Option[DenseMatrix[Double]] = {
    val (nCalled, gtSum, allHet) = gts.flatten.foldLeft((0, 0, true))((acc, gt) => (acc._1 + 1, acc._2 + gt, acc._3 && (gt == 1) ))

    // allHomRef || allHet || allHomVar || allNoCall
    if (gtSum == 0 || allHet || gtSum == 2 * nCalled || nCalled == 0 )
      None
    else {
      val gtMean = gtSum.toDouble / nCalled
      val gtArray = gts.map(_.map(_.toDouble).getOrElse(gtMean)).toArray
      Some(new DenseMatrix(gtArray.length, 1, gtArray))
    }
  }
}

case class LogisticRegression(rdd: RDD[(Variant, Annotation)], `type`: Type)
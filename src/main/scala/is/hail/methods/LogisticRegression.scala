package is.hail.methods

import breeze.linalg._
import org.apache.spark.rdd.RDD
import is.hail.utils._
import is.hail.annotations.Annotation
import is.hail.stats._
import is.hail.variant._
import is.hail.expr._

object LogisticRegression {

  def apply(vds: VariantDataset, y: DenseVector[Double], cov: DenseMatrix[Double], logRegTest: LogisticRegressionTest): LogisticRegression = {
    require(cov.rows == y.size)
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

    val sc = vds.sparkContext
    val yBc = sc.broadcast(y)
    val CBc = sc.broadcast(cov)
    val nullFitBc = sc.broadcast(nullFit)
    val logRegTestBc = sc.broadcast(logRegTest) // Is this worth broadcasting?

    val rdd = vds.rdd.map { case (v, (_, gs)) =>
      (v, buildGtColumn(gs)
          .map { gts =>
            val X = DenseMatrix.horzcat(gts, CBc.value)
            logRegTestBc.value.test(X, yBc.value, nullFitBc.value).toAnnotation
          }
          .getOrElse(Annotation.empty)
      )
    }

    new LogisticRegression(rdd, logRegTest.`type`)
  }

  def buildGtColumn(gs: Iterable[Genotype]): Option[DenseMatrix[Double]] = {
    val (nCalled, gtSum, allHet) = gs.flatMap(_.gt).foldLeft((0, 0, true))((acc, gt) => (acc._1 + 1, acc._2 + gt, acc._3 && (gt == 1) ))

    // allHomRef || allHet || allHomVar || allNoCall
    if (gtSum == 0 || allHet || gtSum == 2 * nCalled || nCalled == 0 )
      None
    else {
      val gtMean = gtSum.toDouble / nCalled
      val gtArray = gs.map(_.gt.map(_.toDouble).getOrElse(gtMean)).toArray
      Some(new DenseMatrix(gtArray.length, 1, gtArray))
    }
  }
}

case class LogisticRegression(rdd: RDD[(Variant, Annotation)], `type`: Type)
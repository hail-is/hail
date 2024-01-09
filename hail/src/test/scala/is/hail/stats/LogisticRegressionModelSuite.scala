package is.hail.stats

import is.hail.HailSuite
import is.hail.utils._

import scala.language.postfixOps
import scala.sys.process._

import breeze.linalg._
import org.testng.SkipException
import org.testng.annotations.Test

class LogisticRegressionModelSuite extends HailSuite {

  @Test def covariatesVsInterceptOnlyTest() {

    /* R code:
     * y0 = c(0, 0, 1, 1, 1, 1) c1 = c(0, 2, 1, -2, -2, 4) c2 = c(-1, 3, 5, 0, -4, 3) logfit <-
     * glm(y0 ~ c1 + c2, family=binomial(link="logit")) summary(logfit)["coefficients"] */

    val y = DenseVector(0d, 0d, 1d, 1d, 1d, 1d)

    val C = DenseMatrix(
      (0.0, -1.0),
      (2.0, 3.0),
      (1.0, 5.0),
      (-2.0, 0.0),
      (-2.0, -4.0),
      (4.0, 3.0),
    )

    val ones = DenseMatrix.fill[Double](6, 1)(1d)

    val nullModel = new LogisticRegressionModel(ones, y)
    val nullFit = nullModel.fit(None, 25, 1e-6)

    val X = DenseMatrix.horzcat(ones, C)

    val waldStats = WaldTest.test(X, y, nullFit, "logistic", 25, 1e-5).stats.get
    val lrStats = LikelihoodRatioTest.test(X, y, nullFit, "logistic", 25, 1e-5).stats.get
    val scoreStats = LogisticScoreTest.test(X, y, nullFit, "logistic", 25, 1e-5).stats.get

    assert(D_==(waldStats.b(0), 0.7245034, tolerance = 1.0e-6))
    assert(D_==(waldStats.b(1), -0.3585773, tolerance = 1.0e-6))
    assert(D_==(waldStats.b(2), 0.1922622, tolerance = 1.0e-6))

    assert(D_==(waldStats.se(0), 0.9396654, tolerance = 1.0e-6))
    assert(D_==(waldStats.se(1), 0.6246568, tolerance = 1.0e-6))
    assert(D_==(waldStats.se(2), 0.4559844, tolerance = 1.0e-6))

    assert(D_==(waldStats.z(0), 0.7710228, tolerance = 1.0e-6))
    assert(D_==(waldStats.z(1), -0.5740389, tolerance = 1.0e-6))
    assert(D_==(waldStats.z(2), 0.4216421, tolerance = 1.0e-5))

    assert(D_==(waldStats.p(0), 0.4406934, tolerance = 1.0e-6))
    assert(D_==(waldStats.p(1), 0.5659415, tolerance = 1.0e-6))
    assert(D_==(waldStats.p(2), 0.6732863, tolerance = 1.0e-6))

    assert(D_==(nullFit.logLkhd, -3.81908501, tolerance = 1.0e-6))
    assert(D_==(lrStats.chi2, 0.351536062, tolerance = 1.0e-6))
    assert(D_==(lrStats.p, 0.8388125392, tolerance = 1.0e-5))

    assert(D_==(scoreStats.chi2, 0.346648, tolerance = 1.0e-5))
    assert(D_==(scoreStats.p, 0.8408652791, tolerance = 1.0e-5))
  }

  @Test def gtsAndCovariatesVsCovariatesOnlyTest() {

    /* R code:
     * y0 <- c(0, 0, 1, 1, 1, 1) c1 <- c(0, 2, 1, -2, -2, 4) c2 <- c(-1, 3, 5, 0, -4, 3) gts <- c(0,
     * 1, 2, 0, 0, 1) logfit <- glm(y0 ~ c1 + c2 + gts, family=binomial(link="logit"))
     * coef(summary(logfit)) */

    val y = DenseVector(0d, 0d, 1d, 1d, 1d, 1d)

    val C = DenseMatrix(
      (1.0, 0.0, -1.0),
      (1.0, 2.0, 3.0),
      (1.0, 1.0, 5.0),
      (1.0, -2.0, 0.0),
      (1.0, -2.0, -4.0),
      (1.0, 4.0, 3.0),
    )

    val gts = DenseVector(0d, 1d, 2d, 0d, 0d, 1d)

    val nullModel = new LogisticRegressionModel(C, y)
    val nullFit = nullModel.fit(None, 25, 1e-6)

    val X = DenseMatrix.horzcat(C, gts.asDenseMatrix.t)

    val waldStats = WaldTest.test(X, y, nullFit, "logistic", 25, 1e-5).stats.get
    val lrStats = LikelihoodRatioTest.test(X, y, nullFit, "logistic", 25, 1e-5).stats.get
    val scoreStats = LogisticScoreTest.test(X, y, nullFit, "logistic", 25, 1e-5).stats.get

    assert(D_==(waldStats.b(0), -0.4811418, tolerance = 1.0e-6))
    assert(D_==(waldStats.b(1), -0.4293547, tolerance = 1.0e-6))
    assert(D_==(waldStats.b(2), -0.4214875, tolerance = 1.0e-6))
    assert(D_==(waldStats.b(3), 3.1775729, tolerance = 1.0e-6))

    assert(D_==(waldStats.se(0), 1.6203668, tolerance = 1.0e-6))
    assert(D_==(waldStats.se(1), 0.7256665, tolerance = 1.0e-6))
    assert(D_==(waldStats.se(2), 0.9223569, tolerance = 1.0e-6))
    assert(D_==(waldStats.se(3), 4.1094207, tolerance = 1.0e-6))

    assert(D_==(waldStats.z(0), -0.2969339, tolerance = 1.0e-6))
    assert(D_==(waldStats.z(1), -0.5916695, tolerance = 1.0e-6))
    assert(D_==(waldStats.z(2), -0.4569679, tolerance = 1.0e-6))
    assert(D_==(waldStats.z(3), 0.7732411, tolerance = 1.0e-6))

    assert(D_==(waldStats.p(0), 0.7665170, tolerance = 1.0e-6))
    assert(D_==(waldStats.p(1), 0.5540720, tolerance = 1.0e-6))
    assert(D_==(waldStats.p(2), 0.6476941, tolerance = 1.0e-6))
    assert(D_==(waldStats.p(3), 0.4393797, tolerance = 1.0e-6))

    /* R code:
     * logfitnull <- glm(y0 ~ c1 + c2, family=binomial(link="logit")) lrtest <- anova(logfitnull,
     * logfit, test="LRT") chi2 <- lrtest[["Deviance"]][2] pval <- lrtest[["Pr(>Chi)"]][2] */

    assert(D_==(nullFit.logLkhd, -3.643316979, tolerance = 1.0e-6))
    assert(D_==(lrStats.chi2, 0.8733246, tolerance = 1.0e-6))
    assert(D_==(lrStats.p, 0.3500365602, tolerance = 1.0e-6))

    /* R code:
     * scoretest <- anova(logfitnull, logfit, test="Rao") chi2 <- scoretest[["Rao"]][2] pval <-
     * scoretest[["Pr(>Chi)"]][2] */

    assert(D_==(scoreStats.chi2, 0.7955342582, tolerance = 1.0e-5))
    assert(D_==(scoreStats.p, 0.3724319159, tolerance = 1.0e-5))
  }

  @Test def firthSeparationTest() {
    val y = DenseVector(0d, 0d, 0d, 1d, 1d, 1d)
    val X = y.asDenseMatrix.t

    val nullModel = new LogisticRegressionModel(X, y) // separation => MLE does not exist
    var nullFit = nullModel.fit(None, 25, 1e-6)
    assert(!nullFit.converged)

    nullFit = GLMFit(
      nullModel.bInterceptOnly(),
      None,
      None,
      0,
      nullFit.nIter,
      exploded = nullFit.exploded,
      converged = false,
    )
    assert(LogisticFirthTest.test(X, y, nullFit, "logistic", 25,
      1e-5).stats.isDefined) // firth penalized MLE still exists
  }
}

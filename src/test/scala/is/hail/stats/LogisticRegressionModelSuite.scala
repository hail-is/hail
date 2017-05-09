package is.hail.stats

import breeze.linalg._
import is.hail.SparkSuite
import is.hail.utils._
import org.testng.annotations.Test

import scala.language.postfixOps
import scala.sys.process._

class LogisticRegressionModelSuite extends SparkSuite {

  @Test def covariatesVsInterceptOnlyTest() {

    /* R code:
    y0 = c(0, 0, 1, 1, 1, 1)
    c1 = c(0, 2, 1, -2, -2, 4)
    c2 = c(-1, 3, 5, 0, -4, 3)
    logfit <- glm(y0 ~ c1 + c2, family=binomial(link="logit"))
    summary(logfit)["coefficients"]
    */

    val y = DenseVector(0d, 0d, 1d, 1d, 1d, 1d)

    val C = DenseMatrix(
      (0.0, -1.0),
      (2.0, 3.0),
      (1.0, 5.0),
      (-2.0, 0.0),
      (-2.0, -4.0),
      (4.0, 3.0))

    val ones = DenseMatrix.fill[Double](6, 1)(1d)

    val nullModel = new LogisticRegressionModel(ones, y)
    val nullFit = nullModel.fit()

    val X = DenseMatrix.horzcat(ones, C)

    val waldStats = WaldTest.test(X, y, nullFit).stats.get
    val lrStats = LikelihoodRatioTest.test(X, y, nullFit).stats.get
    val scoreStats = ScoreTest.test(X, y, nullFit).stats.get

    assert(D_==(waldStats.b(0), 0.7245034, tolerance = 1.0E-6))
    assert(D_==(waldStats.b(1), -0.3585773, tolerance = 1.0E-6))
    assert(D_==(waldStats.b(2), 0.1922622, tolerance = 1.0E-6))

    assert(D_==(waldStats.se(0), 0.9396654, tolerance = 1.0E-6))
    assert(D_==(waldStats.se(1), 0.6246568, tolerance = 1.0E-6))
    assert(D_==(waldStats.se(2), 0.4559844, tolerance = 1.0E-6))

    assert(D_==(waldStats.z(0), 0.7710228, tolerance = 1.0E-6))
    assert(D_==(waldStats.z(1), -0.5740389, tolerance = 1.0E-6))
    assert(D_==(waldStats.z(2), 0.4216421, tolerance = 1.0E-5))

    assert(D_==(waldStats.p(0), 0.4406934, tolerance = 1.0E-6))
    assert(D_==(waldStats.p(1), 0.5659415, tolerance = 1.0E-6))
    assert(D_==(waldStats.p(2), 0.6732863, tolerance = 1.0E-6))

    assert(D_==(nullFit.logLkhd, -3.81908501, tolerance = 1.0E-6))
    assert(D_==(lrStats.chi2, 0.351536062, tolerance = 1.0E-6))
    assert(D_==(lrStats.p, 0.8388125392, tolerance = 1.0E-5))

    assert(D_==(scoreStats.chi2, 0.346648, tolerance = 1.0E-5))
    assert(D_==(scoreStats.p, 0.8408652791, tolerance = 1.0E-5))
  }


  @Test def gtsAndCovariatesVsCovariatesOnlyTest() {

    /* R code:
    y0 <- c(0, 0, 1, 1, 1, 1)
    c1 <- c(0, 2, 1, -2, -2, 4)
    c2 <- c(-1, 3, 5, 0, -4, 3)
    gts <- c(0, 1, 2, 0, 0, 1)
    logfit <- glm(y0 ~ c1 + c2 + gts, family=binomial(link="logit"))
    coef(summary(logfit))
    */

    val y = DenseVector(0d, 0d, 1d, 1d, 1d, 1d)

    val C = DenseMatrix(
      (1.0, 0.0, -1.0),
      (1.0, 2.0, 3.0),
      (1.0, 1.0, 5.0),
      (1.0, -2.0, 0.0),
      (1.0, -2.0, -4.0),
      (1.0, 4.0, 3.0))

    val gts = DenseVector(0d, 1d, 2d, 0d, 0d, 1d)

    val nullModel = new LogisticRegressionModel(C, y)
    val nullFit = nullModel.fit()

    val X = DenseMatrix.horzcat(C, gts.asDenseMatrix.t)

    val waldStats = WaldTest.test(X, y, nullFit).stats.get
    val lrStats = LikelihoodRatioTest.test(X, y, nullFit).stats.get
    val scoreStats = ScoreTest.test(X, y, nullFit).stats.get

    assert(D_==(waldStats.b(0), -0.4811418, tolerance = 1.0E-6))
    assert(D_==(waldStats.b(1), -0.4293547, tolerance = 1.0E-6))
    assert(D_==(waldStats.b(2), -0.4214875, tolerance = 1.0E-6))
    assert(D_==(waldStats.b(3), 3.1775729, tolerance = 1.0E-6))

    assert(D_==(waldStats.se(0), 1.6203668, tolerance = 1.0E-6))
    assert(D_==(waldStats.se(1), 0.7256665, tolerance = 1.0E-6))
    assert(D_==(waldStats.se(2), 0.9223569, tolerance = 1.0E-6))
    assert(D_==(waldStats.se(3), 4.1094207, tolerance = 1.0E-6))

    assert(D_==(waldStats.z(0), -0.2969339, tolerance = 1.0E-6))
    assert(D_==(waldStats.z(1), -0.5916695, tolerance = 1.0E-6))
    assert(D_==(waldStats.z(2), -0.4569679, tolerance = 1.0E-6))
    assert(D_==(waldStats.z(3), 0.7732411, tolerance = 1.0E-6))

    assert(D_==(waldStats.p(0), 0.7665170, tolerance = 1.0E-6))
    assert(D_==(waldStats.p(1), 0.5540720, tolerance = 1.0E-6))
    assert(D_==(waldStats.p(2), 0.6476941, tolerance = 1.0E-6))
    assert(D_==(waldStats.p(3), 0.4393797, tolerance = 1.0E-6))

    /* R code:
    logfitnull <- glm(y0 ~ c1 + c2, family=binomial(link="logit"))
    lrtest <- anova(logfitnull, logfit, test="LRT")
    chi2 <- lrtest[["Deviance"]][2]
    pval <- lrtest[["Pr(>Chi)"]][2]
    */

    assert(D_==(nullFit.logLkhd, -3.643316979, tolerance = 1.0E-6))
    assert(D_==(lrStats.chi2, 0.8733246, tolerance = 1.0E-6))
    assert(D_==(lrStats.p, 0.3500365602, tolerance = 1.0E-6))

    /* R code:
    scoretest <- anova(logfitnull, logfit, test="Rao")
    chi2 <- scoretest[["Rao"]][2]
    pval <- scoretest[["Pr(>Chi)"]][2]
    */

    assert(D_==(scoreStats.chi2, 0.7955342582, tolerance = 1.0E-5))
    assert(D_==(scoreStats.p, 0.3724319159, tolerance = 1.0E-5))
  }

  @Test def covariatesVsInterceptOnlyR() {

    val tol = 1E-5

    def assertArray(v: DenseVector[Double], a: Array[Double], tolerance: Double = tol) {
      assert(v.length == a.length)
      (v.data, a).zipped.foreach((ai, bi) => assert(D_==(ai, bi, tolerance)))
    }

    def readResults(file: String) = {
      hadoopConf.readLines(file) {
        _.map {
          _.map { _.split("""\t""").map(_.toDouble) }.value
        }.toArray
      }
    }

    def jsonStr[T](arr: Array[T]) = arr.mkString("[ ", ", ", " ]")

    val rows = 6
    val cols = 2
    val XArray = Array[Double](0, 2, 1, -2, -2, 4, -1, 3, 5, 0, -4, 3)
    val yArray = Array[Double](0, 0, 1, 1, 1, 1)

    val jsonString =
      s"""{"rows": $rows,
          |"cols": $cols,
          |"X": ${ jsonStr(XArray) },
          |"y": ${ jsonStr(yArray) }
          |}""".stripMargin

    val inputFile = tmpDir.createLocalTempFile("localInput", ".json")
    val waldFile = tmpDir.createLocalTempFile("localWald", ".tsv")
    val lrtFile = tmpDir.createLocalTempFile("localLRT", ".tsv")
    val scoreFile = tmpDir.createLocalTempFile("localScore", ".tsv")
    val firthFitFile = tmpDir.createLocalTempFile("firthfit", ".tsv")
    val firthLrtFile = tmpDir.createLocalTempFile("firthlrt", ".tsv")

    hadoopConf.writeTextFile(inputFile) { _.write(jsonString) }

    val rScript = s"Rscript src/test/resources/regressionLogistic.R " +
      s"${ uriPath(inputFile) } " + s"${ uriPath(waldFile) } ${ uriPath(lrtFile) } " +
      s"${ uriPath(scoreFile) } ${ uriPath(firthFitFile) } ${ uriPath(firthLrtFile) }"

    val exitCode = rScript !

    assert(exitCode == 0, "R process for regressionLogistic.R failed.")

    val waldR = readResults(waldFile)
    val lrtR = readResults(lrtFile)
    val scoreR = readResults(scoreFile)
    val firthFitR = readResults(firthFitFile)
    val firthLrtR = readResults(firthLrtFile)

    val waldBR = waldR.map(_ (0))
    val waldSeR = waldR.map(_ (1))
    val waldZR = waldR.map(_ (2))
    val waldPR = waldR.map(_ (3))

    val lrtChi2R = lrtR(0)(0)
    val lrtPR = lrtR(1)(0)

    val scoreChi2R = scoreR(0)(0)
    val scorePR = scoreR(1)(0)

    val firthBR = firthFitR.map(_(0))
    val firthLogLkhd0R = firthLrtR(0)(0)
    val firthLogLkhdR = firthLrtR(1)(0)
    val firthChi2R = 2 * (firthLogLkhdR - firthLogLkhd0R)
    val firthPValR = firthLrtR(2)(0)

    val y = DenseVector(yArray)
    val C = new DenseMatrix(rows, cols, XArray)
    val ones = DenseMatrix.fill[Double](6, 1)(1d)

    val nullModel = new LogisticRegressionModel(ones, y)
    val nullFit = nullModel.fit()

    val X = DenseMatrix.horzcat(ones, C)

    val waldStats = WaldTest.test(X, y, nullFit).stats.get
    val lrStats = LikelihoodRatioTest.test(X, y, nullFit).stats.get
    val scoreStats = ScoreTest.test(X, y, nullFit).stats.get
    val firthTest = FirthTest.test(X, y, nullFit)
    val firthStats = firthTest.stats.get
    val firthFit = firthTest.fitStats

    assertArray(waldStats.b, waldBR)
    assertArray(waldStats.se, waldSeR)
    assertArray(waldStats.z, waldZR)
    assertArray(waldStats.p, waldPR)

    assert(D_==(lrStats.chi2, lrtChi2R, tol))
    assert(D_==(lrStats.p, lrtPR, tol))

    assert(D_==(scoreStats.chi2, scoreChi2R, tol))
    assert(D_==(scoreStats.p, scorePR, tol))

    assertArray(firthStats.b, firthBR)
    assert(D_==(firthFit.logLkhd, firthLogLkhdR, tol))
    assert(D_==(firthStats.p, firthPValR, tol))
  }
}
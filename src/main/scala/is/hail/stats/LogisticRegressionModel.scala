package is.hail.stats

import breeze.linalg._
import breeze.numerics._
import is.hail.annotations.Annotation
import is.hail.expr._

abstract class LogisticRegressionTest extends Serializable {
  def test(X: DenseMatrix[Double], y: DenseVector[Double], nullFit: LogisticRegressionFit): LogisticRegressionTestResult[LogisticRegressionStats]
  def `type`: Type
}

abstract class LogisticRegressionStats {
  def toAnnotation: Annotation
}

class LogisticRegressionTestResult[+T <: LogisticRegressionStats](val stats: Option[T]) {
  def toAnnotation: Annotation = Annotation(stats.fold(Annotation.empty)(_.toAnnotation))
}

class LogisticRegressionTestResultWithFit[T <: LogisticRegressionStats](override val stats: Option[T], val fitStats: LogisticRegressionFit) extends LogisticRegressionTestResult[T](stats) {
  override def toAnnotation: Annotation = Annotation(stats.fold(Annotation.empty)(_.toAnnotation), fitStats.toAnnotation)
}


object WaldTest extends LogisticRegressionTest {
  def test(X: DenseMatrix[Double], y: DenseVector[Double], nullFit: LogisticRegressionFit): LogisticRegressionTestResultWithFit[WaldStats] = {
    val model = new LogisticRegressionModel(X, y)
    val fit = model.fit(nullFit.b)

    val waldStats = if (fit.converged) {
      try {
        val se = sqrt(diag(inv(fit.fisher)))
        val z = fit.b :/ se
        val p = z.map(zi => chiSquaredTail(1, zi * zi))

        Some(new WaldStats(fit.b, se, z, p))
      } catch {
        case e: breeze.linalg.MatrixSingularException => None
        case e: breeze.linalg.NotConvergedException => None
      }
    } else
      None

    new LogisticRegressionTestResultWithFit[WaldStats](waldStats, fit)
  }

  def `type`: Type = TStruct(
    ("wald", WaldStats.`type`),
    ("fit", LogisticRegressionFit.`type`))
}

object WaldStats {
  def `type`: Type = TStruct(
    ("beta", TDouble),
    ("se", TDouble),
    ("zstat", TDouble),
    ("pval", TDouble))
}

case class WaldStats(b: DenseVector[Double], se: DenseVector[Double], z: DenseVector[Double], p: DenseVector[Double]) extends LogisticRegressionStats {
  def toAnnotation: Annotation = Annotation(b(0), se(0), z(0), p(0))
}



object LikelihoodRatioTest extends LogisticRegressionTest {
  def test(X: DenseMatrix[Double], y: DenseVector[Double], nullFit: LogisticRegressionFit): LogisticRegressionTestResultWithFit[LikelihoodRatioStats] = {
    val model = new LogisticRegressionModel(X, y)
    val fit = model.fit(nullFit.b)

    val lrStats =
      if (fit.converged) {
        val chi2 = 2 * (fit.logLkhd - nullFit.logLkhd)
        val p = chiSquaredTail(nullFit.df, chi2)

        Some(new LikelihoodRatioStats(fit.b, chi2, p))
      } else
        None

    new LogisticRegressionTestResultWithFit[LikelihoodRatioStats](lrStats, fit)
  }

  def `type` = TStruct(
    ("lrt", LikelihoodRatioStats.`type`),
    ("fit", LogisticRegressionFit.`type`))
}

object LikelihoodRatioStats {
  def `type`: Type = TStruct(
    ("beta", TDouble),
    ("chi2", TDouble),
    ("pval", TDouble))
}

case class LikelihoodRatioStats(b: DenseVector[Double], chi2: Double, p: Double) extends LogisticRegressionStats {
  def toAnnotation = Annotation(b(0), chi2, p)
}


object FirthTest extends LogisticRegressionTest {
  def test(X: DenseMatrix[Double], y: DenseVector[Double], nullFit: LogisticRegressionFit): LogisticRegressionTestResultWithFit[LikelihoodRatioStats] = {
    val model = new LogisticRegressionModel(X, y)
    val fit = model.fit(nullFit.b)

    val firthStats =
      if (fit.converged) {
        val chi2 = 2 * (fit.logLkhd - nullFit.logLkhd)
        val p = chiSquaredTail(nullFit.df, chi2)

        Some(new FirthStats(fit.b, chi2, p))
      } else
        None

    new LogisticRegressionTestResultWithFit[FirthStats](firthStats, fit)
  }

  def `type` = TStruct(
    ("firth", LikelihoodRatioStats.`type`),
    ("fit", LogisticRegressionFit.`type`))
}

object FirthStats {
  def `type`: Type = TStruct(
    ("beta", TDouble),
    ("chi2", TDouble),
    ("pval", TDouble))
}

case class FirthStats(b: DenseVector[Double], chi2: Double, p: Double) extends LogisticRegressionStats {
  def toAnnotation = Annotation(b(0), chi2, p)
}


object ScoreTest extends LogisticRegressionTest {
  def test(X: DenseMatrix[Double], y: DenseVector[Double], nullFit: LogisticRegressionFit): LogisticRegressionTestResult[ScoreStats] = {
    val scoreStats = {
      try {
        val mu = sigmoid(X * nullFit.b)

        val r0 = 0 until nullFit.df
        val r1 = nullFit.df to -1

        val X0 = X(::, r0)
        val X1 = X(::, r1)

        val score = nullFit.score.copy
        score(r0) := X0.t * (y - mu)

        val fisher = nullFit.fisher.copy
        fisher(r0, r0) := X0.t * (X0(::, *) :* (mu :* (1d - mu)))
        fisher(r0, r1) := X0.t * (X1(::, *) :* (mu :* (1d - mu)))
        fisher(r1, r0) := fisher(r0, r1).t

        val chi2 = score dot (fisher \ score)
        val p = chiSquaredTail(nullFit.df, chi2)

        Some(new ScoreStats(chi2, p))
      } catch {
        case e: breeze.linalg.MatrixSingularException => None
        case e: breeze.linalg.NotConvergedException => None
      }
    }

    new LogisticRegressionTestResult[ScoreStats](scoreStats)
  }

  def `type`: Type = TStruct(
    ("score", ScoreStats.`type`))
}

object ScoreStats {
  def `type`: Type = TStruct(
    ("chi2", TDouble),
    ("pval", TDouble))
}

case class ScoreStats(chi2: Double, p: Double) extends LogisticRegressionStats {
  def toAnnotation = Annotation(chi2, p)
}



class LogisticRegressionModel(X: DenseMatrix[Double], y: DenseVector[Double]) {
  require(y.length == X.rows)
  require(y.forall(yi => yi == 0 || yi == 1))
  require{ val sumY = sum(y); sumY > 0 && sumY < y.length }

  val n = X.rows
  val m = X.cols

  def bInterceptOnly(interceptCol: Int = 0): DenseVector[Double] = {
    val b = DenseVector.zeros[Double](m)
    val avg = sum(y) / n
    b(interceptCol) = math.log(avg / (1 - avg))
    b
  }

  def fit(b0: DenseVector[Double], maxIter: Int = 25, tol: Double = 1E-6): LogisticRegressionFit = {
    require(X.cols == b0.length)
    require(maxIter > 0)

    var b = b0.copy
    var deltaB = DenseVector.zeros[Double](m)
    var mu = DenseVector.zeros[Double](n)
    var score = DenseVector.zeros[Double](m)
    var fisher = DenseMatrix.zeros[Double](m, m)
    var logLkhd = 0d
    var iter = 0
    var converged = false
    var exploded = false

    while (!converged && !exploded && iter < maxIter) {
      try {
        iter += 1
        mu = sigmoid(X * b)
        score = X.t * (y - mu)
        fisher = X.t * (X(::, *) :* (mu :* (1d - mu)))

        if (max(abs(deltaB)) < tol && iter > 1) {
          converged = true
          logLkhd = sum(breeze.numerics.log((y :* mu) + ((1d - y) :* (1d - mu))))
        } else {
          deltaB = fisher \ score
          b += deltaB
        }
      } catch {
        case e: breeze.linalg.MatrixSingularException => exploded = true
        case e: breeze.linalg.NotConvergedException => exploded = true
      }
    }

    LogisticRegressionFit(b, score, fisher, logLkhd, 0, iter, converged, exploded)
  }

  def fitFirth(b0: DenseVector[Double], colsToFit: Int = m, maxIter: Int = 25, tol: Double = 1E-6): LogisticRegressionFit = {
    require(X.cols == b0.length)
    require(maxIter > 0)

    var b = b0.copy
    var deltaB = DenseVector.zeros[Double](m)
    var mu = DenseVector.zeros[Double](n)
    var score = DenseVector.zeros[Double](m)
    var fisher = DenseMatrix.zeros[Double](m, m)
    var logLkhd = 0d
    var iter = 0
    var converged = false
    var exploded = false

    while (!converged && !exploded && iter < maxIter) {
      try {
        iter += 1

        val X0 = X(::, 0 until colsToFit)
        val mu = sigmoid(X0 * b)
        val XsqrtW = X(::,*) :* sqrt(mu :* (1d - mu))
        val QR = qr.reduced(XsqrtW)
        val sqrtH = norm(QR.q(*, ::)) // FIXME: implement norm2
        val score = X0.t * (y - mu + (sqrtH :* sqrtH :* (.5 - mu)))
        val r0 = QR.r(::, 0 until colsToFit)
        val fisher = r0.t * r0

        if (max(abs(deltaB)) < tol && iter > 1) {
          converged = true
          logLkhd = sum(breeze.numerics.log((y :* mu) + ((1d - y) :* (1d - mu)))) +
            sum(breeze.numerics.log(diag(cholesky(fisher)))) // FIXME: change to logdet?
        } else {
          deltaB = fisher \ score
          b += deltaB
        }
      } catch {
        case e: breeze.linalg.MatrixSingularException => exploded = true
        case e: breeze.linalg.NotConvergedException => exploded = true
      }
    }

    LogisticRegressionFit(b, score, fisher, logLkhd, 0, iter, converged, exploded)
  }

  def nullFit(b0: DenseVector[Double] = DenseVector.zeros[Double](m), df: Int = 1, maxIter: Int = 25, tol: Double = 1E-6): LogisticRegressionFit = {
    val m = df + X.cols
    val b = DenseVector.zeros[Double](m)
    val score = DenseVector.zeros[Double](m)
    val fisher = DenseMatrix.zeros[Double](m, m)
    val fit0 = fit(b0, maxIter, tol)

    b(df until m) := fit0.b
    score(df until m) := fit0.score
    fisher(df until m, df until m) := fit0.fisher

    LogisticRegressionFit(b, score, fisher, fit0.logLkhd, df, fit0.nIter, fit0.converged, fit0.exploded)
  }
}

object LogisticRegressionFit {
  def `type`: Type = TStruct(
    ("nIter", TInt),
    ("converged", TBoolean),
    ("exploded", TBoolean))
}

case class LogisticRegressionFit(
  b: DenseVector[Double],
  score: DenseVector[Double],
  fisher: DenseMatrix[Double],
  logLkhd: Double,
  df: Int, // degrees of freedom of full model relative to this fit model (positive for null model, 0 for full model)
  nIter: Int,
  converged: Boolean,
  exploded: Boolean) {

  def toAnnotation: Annotation = Annotation(nIter, converged, exploded)
}
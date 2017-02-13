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

    val nullFitb = DenseVector.zeros[Double](X.cols)
    nullFitb(0 until nullFit.b.length) := nullFit.b

    val fit = model.fit(nullFitb, computeSe=true)

    val waldStats = if (fit.converged) {
      try {
        val se = fit.optSe.get
        val z = fit.b :/ se
        val p = z.map(zi => 2 * pnorm(-math.abs(zi)))

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
  def toAnnotation: Annotation = Annotation(b(-1), se(-1), z(-1), p(-1))
}



object LikelihoodRatioTest extends LogisticRegressionTest {
  def test(X: DenseMatrix[Double], y: DenseVector[Double], nullFit: LogisticRegressionFit): LogisticRegressionTestResultWithFit[LikelihoodRatioStats] = {

    val m = X.cols
    val m0 = nullFit.b.length
    val model = new LogisticRegressionModel(X, y)

    val nullFitb = DenseVector.zeros[Double](m)
    nullFitb(0 until m0) := nullFit.b

    val fit = model.fit(nullFitb, computeLogLkld = true)

    val lrStats =
      if (fit.converged) {
        val chi2 = 2 * (fit.optLogLkhd.get - nullFit.optLogLkhd.get)
        val p = chiSquaredTail(m - m0, chi2)

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
  def toAnnotation = Annotation(b(-1), chi2, p)
}


object FirthTest extends LogisticRegressionTest {
  def test(X: DenseMatrix[Double], y: DenseVector[Double], nullFit: LogisticRegressionFit): LogisticRegressionTestResultWithFit[FirthStats] = {
    val m = X.cols
    val m0 = nullFit.b.length
    val model = new LogisticRegressionModel(X, y)
    val nullFitFirth = model.fitFirth(nullFit.b)

    if (nullFitFirth.converged) {
      val nullFitFirthb = DenseVector.zeros[Double](m)
      nullFitFirthb(0 until m0) := nullFitFirth.b

      val fitFirth = model.fitFirth(nullFitFirthb)

      val firthStats =
        if (fitFirth.converged) {
          val chi2 = 2 * (fitFirth.optLogLkhd.get - nullFitFirth.optLogLkhd.get)
          val p = chiSquaredTail(m - m0, chi2)

          Some(new FirthStats(fitFirth.b, chi2, p))
        } else
          None

      new LogisticRegressionTestResultWithFit[FirthStats](firthStats, fitFirth)
    } else
      new LogisticRegressionTestResultWithFit[FirthStats](None, nullFitFirth)
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
  def toAnnotation = Annotation(b(-1), chi2, p)
}


object ScoreTest extends LogisticRegressionTest {
  def test(X: DenseMatrix[Double], y: DenseVector[Double], nullFit: LogisticRegressionFit): LogisticRegressionTestResult[ScoreStats] = {
    val scoreStats = {
      try {
        val m = X.cols
        val m0 = nullFit.b.length
        val nullFitb = DenseVector.zeros[Double](m)
        nullFitb(0 until m0) := nullFit.b

        val mu = sigmoid(X * nullFitb)

        val r0 = 0 until m0
        val r1 = m0 to -1

        val X0 = X(::, r0)
        val X1 = X(::, r1)

        val score = DenseVector.zeros[Double](m)
        score(r0) := nullFit.optScore.get
        score(r1) := X1.t * (y - mu)

        val fisher = DenseMatrix.zeros[Double](m, m)
        val R = nullFit.optR.get
        fisher(r0, r0) := R.t * R
        fisher(r0, r1) := X0.t * (X1(::, *) :* (mu :* (1d - mu)))
        fisher(r1, r0) := fisher(r0, r1).t
        fisher(r1, r1) := X1.t * (X1(::, *) :* (mu :* (1d - mu)))

        val chi2 = score dot (fisher \ score)
        val p = chiSquaredTail(m - m0, chi2)

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

  def fit(b0: DenseVector[Double], computeScoreR: Boolean = false, computeSe: Boolean = false, computeLogLkld: Boolean = false, maxIter: Int = 25, tol: Double = 1E-6): LogisticRegressionFit = {
    require(X.cols == b0.length)

    var b = b0.copy
    var deltaB = DenseVector.zeros[Double](m)
    var optScore: Option[DenseVector[Double]] = None
    var optR: Option[DenseMatrix[Double]] = None
    var optSe: Option[DenseVector[Double]] = None
    var optLogLkhd: Option[Double] = None
    var iter = 1
    var converged = false
    var exploded = false

    while (!converged && !exploded && iter <= maxIter) {
      try {
        val mu = sigmoid(X * b)
        val sqrtW = sqrt(mu :* (1d - mu))
        val QR = qr.reduced(X(::, *) :* sqrtW)

        deltaB = TriSolve(QR.r, QR.q.t * ((y - mu) :/ sqrtW))

        if (max(abs(deltaB)) < tol) {
          converged = true
          if (computeScoreR) {
            optScore = Some(X.t * (y - mu))
            optR = Some(QR.r)
          }
          if (computeSe) {
            val invR = inv(QR.r)  // could speed up inverting as upper triangular, or avoid altogether as 1 / se(-1) = fit.fisherSqrt(-1, -1)
            optSe = Some(norm(invR(*, ::)))
          }
          if (computeLogLkld)
            optLogLkhd = Some(sum(breeze.numerics.log((y :* mu) + ((1d - y) :* (1d - mu)))))
        } else {
          iter += 1
          b += deltaB
        }
      } catch {
        case e: breeze.linalg.MatrixSingularException => exploded = true; println("sing")
        case e: breeze.linalg.NotConvergedException => exploded = true; println("notconv")
      }
    }

    LogisticRegressionFit(b, optScore, optR, optSe, optLogLkhd, iter, converged, exploded)
  }

  def fitFirth(b0: DenseVector[Double], maxIter: Int = 100, tol: Double = 1E-6): LogisticRegressionFit = {
    require(b0.length <= m)

    var b = b0.copy
    val m0 = b0.length
    var logLkhd = 0d
    var iter = 1
    var converged = false
    var exploded = false

    var deltaB = DenseVector.zeros[Double](m0)

    while (!converged && !exploded && iter <= maxIter) {
      try {
        val mu = sigmoid(X(::, 0 until m0) * b)
        val sqrtW = sqrt(mu :* (1d - mu))
        val QR = qr.reduced(X(::, *) :* sqrtW)
        val sqrtH = norm(QR.q(*, ::))
        deltaB = TriSolve(QR.r(0 until m0, 0 until m0), QR.q(::, 0 until m0).t * ((y - mu) + (sqrtH :* sqrtH :* (0.5 - mu)) :/ sqrtW))

        if (max(abs(deltaB)) < tol && iter > 1) {
          converged = true
          logLkhd = sum(breeze.numerics.log((y :* mu) + ((1d - y) :* (1d - mu)))) + sum(log(abs(diag(QR.r))))
        } else {
          iter += 1
          b += deltaB
        }
      } catch {
        case e: breeze.linalg.MatrixSingularException => exploded = true
        case e: breeze.linalg.NotConvergedException => exploded = true
      }
    }

    LogisticRegressionFit(b, None, None, None, Some(logLkhd), iter, converged, exploded)
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
  optScore: Option[DenseVector[Double]],
  optR: Option[DenseMatrix[Double]],
  optSe: Option[DenseVector[Double]],
  optLogLkhd: Option[Double],
  nIter: Int,
  converged: Boolean,
  exploded: Boolean) {

  def toAnnotation: Annotation = Annotation(nIter, converged, exploded)
}

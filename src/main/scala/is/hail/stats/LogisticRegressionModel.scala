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

    val fit = model.fit(nullFitb)

    val waldStats = if (fit.converged) {
      try {
        val se = sqrt(diag(inv(fit.fisher)))
        val z = fit.b :/ se
        val p = z.map(zi => chiSquaredTail(1, zi * zi)) // fixme: change to normal tail?

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

    val fit = model.fit(nullFitb)

    val lrStats =
      if (fit.converged) {
        val chi2 = 2 * (fit.logLkhd - nullFit.logLkhd)
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

    val nullFitFirthb = DenseVector.zeros[Double](m)
    nullFitFirthb(0 until m0) := nullFitFirth.b

    val fitFirth = model.fitFirth(nullFitFirthb)

    val firthStats =
      if (nullFitFirth.converged && fitFirth.converged) {
        val chi2 = 2 * (fitFirth.logLkhd - nullFitFirth.logLkhd)
        val p = chiSquaredTail(m - m0, chi2)

        println(s"nullFitFirthB = ${nullFitFirth.b}")
        println(s"nullFitFirthScore = ${nullFitFirth.score}")
        println(s"nullFitFirthFisher = ${nullFitFirth.fisher}")
        println(s"nullFitFirthLkhd = ${nullFitFirth.logLkhd}")
        println(s"fitFirthLkhd = ${fitFirth.logLkhd}")
        println(s"chi2 = $chi2")
        println(s"p = $p")

        Some(new FirthStats(fitFirth.b, chi2, p))
      } else
        None

    new LogisticRegressionTestResultWithFit[FirthStats](firthStats, fitFirth) // FIXME: modify fit annotations to deal with fitting null as well
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
        score(r0) := nullFit.score
        score(r1) := X1.t * (y - mu)

        val fisher = DenseMatrix.zeros[Double](m, m)
        fisher(r0, r0) := nullFit.fisher
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

  def fit(b0: DenseVector[Double], maxIter: Int = 25, tol: Double = 1E-6): LogisticRegressionFit = {
    require(X.cols == b0.length)

    var b = b0.copy
    var deltaB = DenseVector.zeros[Double](m)
    var score = DenseVector.zeros[Double](m)
    var fisher = DenseMatrix.zeros[Double](m, m)
    var logLkhd = 0d
    var iter = 0
    var converged = false
    var exploded = false

    while (!converged && !exploded && iter < maxIter) {
      try {
        iter += 1
        val mu = sigmoid(X * b)
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

    LogisticRegressionFit(b, score, fisher, logLkhd, iter, converged, exploded)
  }

  def fitFirth(b0: DenseVector[Double], maxIter: Int = 100, tol: Double = 1E-6): LogisticRegressionFit = {
    require(b0.length <= m)

    var b = b0.copy
    val m0 = b0.length
    val X0 = if (m == m0) X else X(::, 0 until m0)

    var score = DenseVector.zeros[Double](m0)
    var fisher = DenseMatrix.zeros[Double](m0, m0)
    var logLkhd = 0d
    var iter = 0
    var converged = false
    var exploded = false

    var deltaB = DenseVector.zeros[Double](m0)

    while (!converged && !exploded && iter < maxIter) {
      try {
        iter += 1

        val mu = sigmoid(X0 * b)
        val XsqrtW = X(::,*) :* sqrt(mu :* (1d - mu))
        val QR = qr.reduced(XsqrtW)
        val sqrtH = norm(QR.q(*, ::)) // FIXME: implement norm2
        score = X0.t * (y - mu + (sqrtH :* sqrtH :* (0.5 - mu)))
        val r = if (m == m0) QR.r else QR.r(::, 0 until m0)
        fisher = r.t * r

        if (max(abs(deltaB)) < tol && iter > 1) {
          converged = true
          logLkhd = sum(breeze.numerics.log((y :* mu) + ((1d - y) :* (1d - mu)))) + 0.5 * breeze.linalg.logdet(fisher)._2
        } else {
          deltaB = fisher \ score
          b += deltaB
        }
      } catch {
        case e: breeze.linalg.MatrixSingularException => exploded = true
        case e: breeze.linalg.NotConvergedException => exploded = true
      }
    }

    LogisticRegressionFit(b, score, fisher, logLkhd, iter, converged, exploded)
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
  nIter: Int,
  converged: Boolean,
  exploded: Boolean) {

  def toAnnotation: Annotation = Annotation(nIter, converged, exploded)
}
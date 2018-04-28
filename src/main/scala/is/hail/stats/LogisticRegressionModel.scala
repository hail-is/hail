package is.hail.stats

import breeze.linalg._
import breeze.numerics._
import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.expr.types._
import is.hail.utils.FastSeq

object LogisticRegressionTest {
  val tests = Map("wald" -> WaldTest, "lrt" -> LikelihoodRatioTest, "score" -> ScoreTest, "firth" -> FirthTest)
}

abstract class LogisticRegressionTest extends Serializable {
  def test(X: DenseMatrix[Double], y: DenseVector[Double], nullFit: LogisticRegressionFit): LogisticRegressionTestResult[LogisticRegressionStats]
  val schema: Type
  val emptyStats: Seq[Annotation]
}

abstract class LogisticRegressionStats {
  def toAnnotations: Seq[Annotation]
}

class LogisticRegressionTestResult[+T <: LogisticRegressionStats](val stats: Option[T], private val emptyStats: Seq[Annotation]) {
  def toAnnotation: Annotation =
    Annotation.fromSeq(stats.map(_.toAnnotations).getOrElse(emptyStats))
}

class LogisticRegressionTestResultWithFit[T <: LogisticRegressionStats](override val stats: Option[T], private val emptyStats: Seq[Annotation], val fitStats: LogisticRegressionFit) extends LogisticRegressionTestResult[T](stats, emptyStats) {
  override def toAnnotation: Annotation =
    Annotation.fromSeq(stats.map(_.toAnnotations).getOrElse(emptyStats) :+ fitStats.toAnnotation)
}


object WaldTest extends LogisticRegressionTest {
  val schema: Type = TStruct(
    ("beta", TFloat64()),
    ("standard_error", TFloat64()),
    ("z_stat", TFloat64()),
    ("p_value", TFloat64()),
    ("fit", LogisticRegressionFit.schema))

  val emptyStats: Seq[Annotation] = new Array[Annotation](schema.children.size - 1)

  def test(X: DenseMatrix[Double], y: DenseVector[Double], nullFit: LogisticRegressionFit): LogisticRegressionTestResultWithFit[WaldStats] = {
    require(nullFit.fisher.isDefined)

    val model = new LogisticRegressionModel(X, y)
    val fit = model.fit(Some(nullFit))

    val waldStats = if (fit.converged) {
      try {
        val se = sqrt(diag(inv(fit.fisher.get)))
        val z = fit.b /:/ se
        val p = z.map(zi => 2 * pnorm(-math.abs(zi)))

        Some(WaldStats(fit.b, se, z, p))
      } catch {
        case e: breeze.linalg.MatrixSingularException => None
        case e: breeze.linalg.NotConvergedException => None
      }
    } else
      None

    new LogisticRegressionTestResultWithFit[WaldStats](waldStats, emptyStats, fit)
  }
}

case class WaldStats(b: DenseVector[Double], se: DenseVector[Double], z: DenseVector[Double], p: DenseVector[Double]) extends LogisticRegressionStats {
  def toAnnotations: Seq[Annotation] = FastSeq(b(-1), se(-1), z(-1), p(-1))
}



object LikelihoodRatioTest extends LogisticRegressionTest {
  val schema = TStruct(
    ("beta", TFloat64()),
    ("chi_sq_stat", TFloat64()),
    ("p_value", TFloat64()),
    ("fit", LogisticRegressionFit.schema))

  val emptyStats: Seq[Annotation] = new Array[Annotation](schema.children.size - 1)

  def test(X: DenseMatrix[Double], y: DenseVector[Double], nullFit: LogisticRegressionFit):
    LogisticRegressionTestResultWithFit[LikelihoodRatioStats] = {
    val m = X.cols
    val m0 = nullFit.b.length
    val model = new LogisticRegressionModel(X, y)
    val fit = model.fit(Some(nullFit))

    val lrStats =
      if (fit.converged) {
        val chi2 = 2 * (fit.logLkhd - nullFit.logLkhd)
        val p = chiSquaredTail(chi2, m - m0)

        Some(LikelihoodRatioStats(fit.b, chi2, p))
      } else
        None

    new LogisticRegressionTestResultWithFit[LikelihoodRatioStats](lrStats, emptyStats, fit)
  }
}



case class LikelihoodRatioStats(b: DenseVector[Double], chi2: Double, p: Double) extends LogisticRegressionStats {
  def toAnnotations: Seq[Annotation] = FastSeq(b(-1), chi2, p)
}


object FirthTest extends LogisticRegressionTest {
  val schema = TStruct(
    ("beta", TFloat64()),
    ("chi_sq_stat", TFloat64()),
    ("p_value", TFloat64()),
    ("fit", LogisticRegressionFit.schema))

  val emptyStats: Seq[Annotation] = new Array[Annotation](schema.children.size - 1)

  def test(X: DenseMatrix[Double], y: DenseVector[Double], nullFit: LogisticRegressionFit):
    LogisticRegressionTestResultWithFit[FirthStats] = {

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
          val chi2 = 2 * (fitFirth.logLkhd - nullFitFirth.logLkhd)
          val p = chiSquaredTail(chi2, m - m0)

          Some(FirthStats(fitFirth.b, chi2, p))
        } else
          None

      new LogisticRegressionTestResultWithFit[FirthStats](firthStats, emptyStats, fitFirth)
    } else
      new LogisticRegressionTestResultWithFit[FirthStats](None, emptyStats, nullFitFirth)
  }
}


case class FirthStats(b: DenseVector[Double], chi2: Double, p: Double) extends LogisticRegressionStats {
  def toAnnotations: Seq[Annotation] = FastSeq(b(-1), chi2, p)
}


object ScoreTest extends LogisticRegressionTest {
  val schema: Type = TStruct(
    ("chi_sq_stat", TFloat64()),
    ("p_value", TFloat64()))

  val emptyStats: Seq[Annotation] = new Array[Annotation](schema.children.size)

  def test(X: DenseMatrix[Double], y: DenseVector[Double], nullFit: LogisticRegressionFit):
    LogisticRegressionTestResult[ScoreStats] = {

    require(nullFit.score.isDefined && nullFit.fisher.isDefined)

    val scoreStats = {
      try {
        val m = X.cols
        val b = DenseVector.zeros[Double](m)
        val score = DenseVector.zeros[Double](m)
        val fisher = DenseMatrix.zeros[Double](m, m)

        val m0 = nullFit.b.length

        val r0 = 0 until m0
        val r1 = m0 to -1

        val X0 = X(::, r0)
        val X1 = X(::, r1)

        b(r0) := nullFit.b
        val mu = sigmoid(X * b)
        score(r0) := nullFit.score.get
        score(r1) := X1.t * (y - mu)
        fisher(r0, r0) := nullFit.fisher.get
        fisher(r0, r1) := X0.t * (X1(::, *) *:* (mu *:* (1d - mu)))
        fisher(r1, r0) := fisher(r0, r1).t
        fisher(r1, r1) := X1.t * (X1(::, *) *:* (mu *:* (1d - mu)))

        val chi2 = score dot (fisher \ score)
        val p = chiSquaredTail(chi2, m - m0)

        Some(ScoreStats(chi2, p))
      } catch {
        case e: breeze.linalg.MatrixSingularException => None
        case e: breeze.linalg.NotConvergedException => None
      }
    }

    new LogisticRegressionTestResult[ScoreStats](scoreStats, emptyStats)
  }
}


case class ScoreStats(chi2: Double, p: Double) extends LogisticRegressionStats {
  def toAnnotations: Seq[Annotation] = FastSeq(chi2, p)
}



class LogisticRegressionModel(X: DenseMatrix[Double], y: DenseVector[Double]) {
  require(y.length == X.rows)
  require(y.forall(yi => yi == 0 || yi == 1))
  require{ val sumY = sum(y); sumY > 0 && sumY < y.length }

  val n: Int = X.rows
  val m: Int = X.cols

  def bInterceptOnly(): DenseVector[Double] = {
    val b = DenseVector.zeros[Double](m)
    val avg = sum(y) / n
    b(0) = math.log(avg / (1 - avg))
    b
  }

  def fit(optNullFit: Option[LogisticRegressionFit] = None, maxIter: Int = 25, tol: Double = 1E-6): LogisticRegressionFit = {

    val b = DenseVector.zeros[Double](m)
    val mu = DenseVector.zeros[Double](n)
    val score = DenseVector.zeros[Double](m)
    val fisher = DenseMatrix.zeros[Double](m, m)

    optNullFit match {
      case None =>
        b := bInterceptOnly()
        mu := sigmoid(X * b)
        score := X.t * (y - mu)
        fisher := X.t * (X(::, *) *:* (mu *:* (1d - mu)))
      case Some(nullFit) =>
        val m0 = nullFit.b.length

        val r0 = 0 until m0
        val r1 = m0 to -1

        val X0 = X(::, r0)
        val X1 = X(::, r1)

        b(r0) := nullFit.b
        mu := sigmoid(X * b)
        score(r0) := nullFit.score.get
        score(r1) := X1.t * (y - mu)
        fisher(r0, r0) := nullFit.fisher.get
        fisher(r0, r1) := X0.t * (X1(::, *) *:* (mu *:* (1d - mu)))
        fisher(r1, r0) := fisher(r0, r1).t
        fisher(r1, r1) := X1.t * (X1(::, *) *:* (mu *:* (1d - mu)))
    }

    var iter = 1
    var converged = false
    var exploded = false

    val deltaB = DenseVector.zeros[Double](m)

    while (!converged && !exploded && iter <= maxIter) {
      try {
        deltaB := fisher \ score

        if (max(abs(deltaB)) < tol) {
          converged = true
        } else {
          iter += 1
          b += deltaB
          mu := sigmoid(X * b)
          score := X.t * (y - mu)
          fisher := X.t * (X(::, *) *:* (mu *:* (1d - mu)))
        }
      } catch {
        case e: breeze.linalg.MatrixSingularException => exploded = true
        case e: breeze.linalg.NotConvergedException => exploded = true
      }
    }

    val logLkhd = sum(breeze.numerics.log((y *:* mu) + ((1d - y) *:* (1d - mu))))

    LogisticRegressionFit(b, Some(score), Some(fisher), logLkhd, iter, converged, exploded)
  }

  def fitFirth(b0: DenseVector[Double], maxIter: Int = 100, tol: Double = 1E-6): LogisticRegressionFit = {
    require(b0.length <= m)

    val b = b0.copy
    val m0 = b0.length
    var logLkhd = 0d
    var iter = 1
    var converged = false
    var exploded = false

    while (!converged && !exploded && iter <= maxIter) {
      try {
        val mu = sigmoid(X(::, 0 until m0) * b)
        val sqrtW = sqrt(mu *:* (1d - mu))
        val QR = qr.reduced(X(::, *) *:* sqrtW)
        val h = QR.q(*, ::).map(r => r dot r)
        val deltaB = TriSolve(QR.r(0 until m0, 0 until m0), QR.q(::, 0 until m0).t * (((y - mu) + (h *:* (0.5 - mu))) /:/ sqrtW))

        if (max(abs(deltaB)) < tol && iter > 1) {
          converged = true
          logLkhd = sum(breeze.numerics.log((y *:* mu) + ((1d - y) *:* (1d - mu)))) + sum(log(abs(diag(QR.r))))
        } else {
          iter += 1
          b += deltaB
        }
      } catch {
        case e: breeze.linalg.MatrixSingularException => exploded = true
        case e: breeze.linalg.NotConvergedException => exploded = true
      }
    }

    LogisticRegressionFit(b, None, None, logLkhd, iter, converged, exploded)
  }
}

object LogisticRegressionFit {
  val schema: Type = TStruct(
    ("n_iterations", TInt32()),
    ("converged", TBoolean()),
    ("exploded", TBoolean()))
}

case class LogisticRegressionFit(
  b: DenseVector[Double],
  score: Option[DenseVector[Double]],
  fisher: Option[DenseMatrix[Double]],
  logLkhd: Double,
  nIter: Int,
  converged: Boolean,
  exploded: Boolean) {

  def toAnnotation: Annotation = Annotation(nIter, converged, exploded)
}

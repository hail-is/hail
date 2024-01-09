package is.hail.stats

import is.hail.annotations.RegionValueBuilder
import is.hail.types.virtual._
import is.hail.utils.fatal

import breeze.linalg._
import breeze.numerics._

object LogisticRegressionTest {
  val tests = Map(
    "wald" -> WaldTest,
    "lrt" -> LikelihoodRatioTest,
    "score" -> LogisticScoreTest,
    "firth" -> LogisticFirthTest,
  )
}

abstract class GLMTest extends Serializable {
  def test(
    X: DenseMatrix[Double],
    y: DenseVector[Double],
    nullFit: GLMFit,
    link: String,
    maxIter: Int,
    tol: Double,
  ): GLMTestResult[GLMStats]

  val schema: TStruct
}

abstract class GLMStats {
  def addToRVB(rvb: RegionValueBuilder)
}

class GLMTestResult[+T <: GLMStats](val stats: Option[T], private val size: Int) {
  def addToRVB(rvb: RegionValueBuilder) {
    stats match {
      case Some(s) => s.addToRVB(rvb)
      case None => rvb.skipFields(size)
    }
  }
}

class GLMTestResultWithFit[T <: GLMStats](
  override val stats: Option[T],
  private val size: Int,
  val fitStats: GLMFit,
) extends GLMTestResult[T](stats, size) {
  override def addToRVB(rvb: RegionValueBuilder) {
    super.addToRVB(rvb)
    fitStats.addToRVB(rvb)
  }
}

object WaldTest extends GLMTest {
  val schema: TStruct = TStruct(
    ("beta", TFloat64),
    ("standard_error", TFloat64),
    ("z_stat", TFloat64),
    ("p_value", TFloat64),
    ("fit", GLMFit.schema),
  )

  def test(
    X: DenseMatrix[Double],
    y: DenseVector[Double],
    nullFit: GLMFit,
    link: String,
    maxIter: Int,
    tol: Double,
  ): GLMTestResultWithFit[WaldStats] = {
    require(nullFit.fisher.isDefined)

    val model = link match {
      case "logistic" => new LogisticRegressionModel(X, y)
      case "poisson" => new PoissonRegressionModel(X, y)
      case _ => fatal("link must be logistic or poisson")
    }
    val fit = model.fit(Some(nullFit), maxIter = maxIter, tol = tol)

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

    new GLMTestResultWithFit[WaldStats](waldStats, schema.size - 1, fit)
  }
}

case class WaldStats(
  b: DenseVector[Double],
  se: DenseVector[Double],
  z: DenseVector[Double],
  p: DenseVector[Double],
) extends GLMStats {
  def addToRVB(rvb: RegionValueBuilder): Unit = {
    rvb.addDouble(b(-1))
    rvb.addDouble(se(-1))
    rvb.addDouble(z(-1))
    rvb.addDouble(p(-1))
  }
}

object LikelihoodRatioTest extends GLMTest {
  val schema = TStruct(
    ("beta", TFloat64),
    ("chi_sq_stat", TFloat64),
    ("p_value", TFloat64),
    ("fit", GLMFit.schema),
  )

  def test(
    X: DenseMatrix[Double],
    y: DenseVector[Double],
    nullFit: GLMFit,
    link: String,
    maxIter: Int,
    tol: Double,
  ): GLMTestResultWithFit[LikelihoodRatioStats] = {
    val m = X.cols
    val m0 = nullFit.b.length
    val model = link match {
      case "logistic" => new LogisticRegressionModel(X, y)
      case "poisson" => new PoissonRegressionModel(X, y)
      case _ => fatal("link must be logistic or poisson")
    }
    val fit = model.fit(Some(nullFit), maxIter = maxIter, tol = tol)

    val lrStats =
      if (fit.converged) {
        val chi2 = 2 * (fit.logLkhd - nullFit.logLkhd)
        val p = pchisqtail(chi2, m - m0)

        Some(LikelihoodRatioStats(fit.b, chi2, p))
      } else
        None

    new GLMTestResultWithFit[LikelihoodRatioStats](lrStats, schema.size - 1, fit)
  }
}

case class LikelihoodRatioStats(b: DenseVector[Double], chi2: Double, p: Double) extends GLMStats {
  def addToRVB(rvb: RegionValueBuilder): Unit = {
    rvb.addDouble(b(-1))
    rvb.addDouble(chi2)
    rvb.addDouble(p)
  }
}

object LogisticFirthTest extends GLMTest {
  val schema = TStruct(
    ("beta", TFloat64),
    ("chi_sq_stat", TFloat64),
    ("p_value", TFloat64),
    ("fit", GLMFit.schema),
  )

  def test(
    X: DenseMatrix[Double],
    y: DenseVector[Double],
    nullFit: GLMFit,
    link: String,
    maxIter: Int,
    tol: Double,
  ): GLMTestResultWithFit[FirthStats] = {
    require(link == "logistic")

    val m = X.cols
    val m0 = nullFit.b.length
    val model = new LogisticRegressionModel(X, y)
    val nullFitFirth = model.fitFirth(nullFit.b, maxIter = maxIter, tol = tol)

    if (nullFitFirth.converged) {
      val nullFitFirthb = DenseVector.zeros[Double](m)
      nullFitFirthb(0 until m0) := nullFitFirth.b

      val fitFirth = model.fitFirth(nullFitFirthb, maxIter = maxIter, tol = tol)

      val firthStats =
        if (fitFirth.converged) {
          val chi2 = 2 * (fitFirth.logLkhd - nullFitFirth.logLkhd)
          val p = pchisqtail(chi2, m - m0)

          Some(FirthStats(fitFirth.b, chi2, p))
        } else
          None

      new GLMTestResultWithFit[FirthStats](firthStats, schema.size - 1, fitFirth)
    } else
      new GLMTestResultWithFit[FirthStats](None, schema.size - 1, nullFitFirth)
  }
}

case class FirthStats(b: DenseVector[Double], chi2: Double, p: Double) extends GLMStats {
  def addToRVB(rvb: RegionValueBuilder): Unit = {
    rvb.addDouble(b(-1))
    rvb.addDouble(chi2)
    rvb.addDouble(p)
  }
}

object LogisticScoreTest extends GLMTest {
  val schema: TStruct = TStruct(
    ("chi_sq_stat", TFloat64),
    ("p_value", TFloat64),
  )

  def test(
    X: DenseMatrix[Double],
    y: DenseVector[Double],
    nullFit: GLMFit,
    link: String,
    maxIter: Int,
    tol: Double,
  ): GLMTestResult[ScoreStats] = {
    require(link == "logistic")
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
        val p = pchisqtail(chi2, m - m0)

        Some(ScoreStats(chi2, p))
      } catch {
        case e: breeze.linalg.MatrixSingularException => None
        case e: breeze.linalg.NotConvergedException => None
      }
    }

    new GLMTestResult[ScoreStats](scoreStats, schema.size)
  }
}

case class ScoreStats(chi2: Double, p: Double) extends GLMStats {
  def addToRVB(rvb: RegionValueBuilder): Unit = {
    rvb.addDouble(chi2)
    rvb.addDouble(p)
  }
}

abstract class GeneralLinearModel {
  def bInterceptOnly(): DenseVector[Double]

  def fit(optNullFit: Option[GLMFit], maxIter: Int, tol: Double): GLMFit
}

class LogisticRegressionModel(X: DenseMatrix[Double], y: DenseVector[Double])
    extends GeneralLinearModel {
  require(y.length == X.rows)

  val n: Int = X.rows
  val m: Int = X.cols

  def bInterceptOnly(): DenseVector[Double] = {
    require(m > 0)
    val b = DenseVector.zeros[Double](m)
    val avg = sum(y) / n
    b(0) = math.log(avg / (1 - avg))
    b
  }

  def fit(optNullFit: Option[GLMFit] = None, maxIter: Int, tol: Double): GLMFit = {

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

    var iter = 0
    var converged = false
    var exploded = false

    val deltaB = DenseVector.zeros[Double](m)

    while (!converged && !exploded && iter < maxIter) {
      iter += 1
      try {
        deltaB := fisher \ score

        if (deltaB(0).isNaN) {
          exploded = true
        } else if (max(abs(deltaB)) < tol) {
          converged = true
        } else {
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

    GLMFit(b, Some(score), Some(fisher), logLkhd, iter, converged, exploded)
  }

  def fitFirth(b0: DenseVector[Double], maxIter: Int, tol: Double): GLMFit = {
    require(b0.length <= m)

    val b = b0.copy
    val m0 = b0.length
    var logLkhd = 0d
    var iter = 0
    var converged = false
    var exploded = false

    while (!converged && !exploded && iter < maxIter) {
      iter += 1
      try {
        val mu = sigmoid(X(::, 0 until m0) * b)
        val sqrtW = sqrt(mu *:* (1d - mu))
        val QR = qr.reduced(X(::, *) *:* sqrtW)
        val h = QR.q(*, ::).map(r => r dot r)
        val deltaB = TriSolve(
          QR.r(0 until m0, 0 until m0),
          QR.q(::, 0 until m0).t * (((y - mu) + (h *:* (0.5 - mu))) /:/ sqrtW),
        )

        if (deltaB(0).isNaN) {
          exploded = true
        } else if (max(abs(deltaB)) < tol && iter > 1) {
          converged = true
          logLkhd = sum(breeze.numerics.log((y *:* mu) + ((1d - y) *:* (1d - mu)))) + sum(
            log(abs(diag(QR.r)))
          )
        } else {
          b += deltaB
        }
      } catch {
        case e: breeze.linalg.MatrixSingularException => exploded = true
        case e: breeze.linalg.NotConvergedException => exploded = true
      }
    }

    GLMFit(b, None, None, logLkhd, iter, converged, exploded)
  }
}

object GLMFit {
  val schema: Type = TStruct(
    ("n_iterations", TInt32),
    ("converged", TBoolean),
    ("exploded", TBoolean),
  )
}

case class GLMFit(
  b: DenseVector[Double],
  score: Option[DenseVector[Double]],
  fisher: Option[DenseMatrix[Double]],
  logLkhd: Double,
  nIter: Int,
  converged: Boolean,
  exploded: Boolean,
) {

  def addToRVB(rvb: RegionValueBuilder): Unit = {
    rvb.startStruct()
    rvb.addInt(nIter)
    rvb.addBoolean(converged)
    rvb.addBoolean(exploded)
    rvb.endStruct()
  }
}

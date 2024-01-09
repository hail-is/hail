package is.hail.stats

import is.hail.types.virtual.{TFloat64, TStruct}

import breeze.linalg._
import breeze.numerics._

object PoissonRegressionTest {
  val tests = Map("wald" -> WaldTest, "lrt" -> LikelihoodRatioTest, "score" -> PoissonScoreTest)
}

object PoissonScoreTest extends GLMTest {
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
    require(link == "poisson")
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
        val mu = exp(X * b)
        score(r0) := nullFit.score.get
        score(r1) := X1.t * (y - mu)
        fisher(r0, r0) := nullFit.fisher.get
        fisher(r0, r1) := X0.t * (X1(::, *) *:* mu)
        fisher(r1, r0) := fisher(r0, r1).t
        fisher(r1, r1) := X1.t * (X1(::, *) *:* mu)

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

class PoissonRegressionModel(X: DenseMatrix[Double], y: DenseVector[Double])
    extends GeneralLinearModel {
  require(y.length == X.rows)

  val n: Int = X.rows
  val m: Int = X.cols

  def bInterceptOnly(): DenseVector[Double] = {
    require(m > 0)
    val b = DenseVector.zeros[Double](m)
    val avg = sum(y) / n
    b(0) = math.log(avg)
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
        mu := exp(X * b)
        score := X.t * (y - mu)
        fisher := X.t * (X(::, *) *:* mu)
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
        fisher(r0, r1) := X0.t * (X1(::, *) *:* mu)
        fisher(r1, r0) := fisher(r0, r1).t
        fisher(r1, r1) := X1.t * (X1(::, *) *:* mu)
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
          mu := exp(X * b)
          score := X.t * (y - mu)
          fisher := X.t * (X(::, *) *:* mu)
        }
      } catch {
        case e: breeze.linalg.MatrixSingularException => exploded = true
        case e: breeze.linalg.NotConvergedException => exploded = true
      }
    }

    /* dropping constant that depends only on y:
     * -sum(breeze.numerics.lgamma(DenseVector(y.data.filter(_ != 0.0)))) */
    val logLkhd = (y dot breeze.numerics.log(mu)) - sum(mu)

    GLMFit(b, Some(score), Some(fisher), logLkhd, iter, converged, exploded)
  }
}

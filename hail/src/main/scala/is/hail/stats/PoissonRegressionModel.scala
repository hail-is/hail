package is.hail.stats

import breeze.linalg._
import breeze.numerics._
import is.hail.annotations.Annotation
import is.hail.expr.types._


object PoissonRegressionTest {
  val tests = Map("wald" -> WaldTest, "lrt" -> LikelihoodRatioTest, "score" -> LogisticScoreTest)
}


object PoissonScoreTest extends GLMTest {
  val schema: Type = TStruct(
    ("chi_sq_stat", TFloat64()),
    ("p_value", TFloat64()))

  val emptyStats: Seq[Annotation] = new Array[Annotation](schema.children.size)

  def test(X: DenseMatrix[Double], y: DenseVector[Double], nullFit: GLMFit):
    GLMTestResult[ScoreStats] = {

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
        val p = chiSquaredTail(chi2, m - m0)

        Some(ScoreStats(chi2, p))
      } catch {
        case e: breeze.linalg.MatrixSingularException => None
        case e: breeze.linalg.NotConvergedException => None
      }
    }

    new GLMTestResult[ScoreStats](scoreStats, emptyStats)
  }
}


class PoissonRegressionModel(X: DenseMatrix[Double], y: DenseVector[Double]) {
  require(y.length == X.rows)
  require(y.forall(yi => math.floor(yi) == yi && yi >= 0))

  val n: Int = X.rows
  val m: Int = X.cols

  def bInterceptOnly(): DenseVector[Double] = {
    require(m > 0)
    val b = DenseVector.zeros[Double](m)
    val avg = sum(y) / n
    b(0) = math.log(avg / (1 - avg))
    b
  }

  def fit(optNullFit: Option[GLMFit] = None, maxIter: Int = 25, tol: Double = 1E-6): GLMFit = {

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
          mu := exp(X * b)
          score := X.t * (y - mu)
          fisher := X.t * (X(::, *) *:* mu)
        }
      } catch {
        case e: breeze.linalg.MatrixSingularException => exploded = true
        case e: breeze.linalg.NotConvergedException => exploded = true
      }
    }

    val logLkhd = sum(breeze.numerics.log(y *:* mu) - y - breeze.numerics.lgamma(y)) // cache lgamma for small counts?

    GLMFit(b, Some(score), Some(fisher), logLkhd, iter, converged, exploded)
  }
}

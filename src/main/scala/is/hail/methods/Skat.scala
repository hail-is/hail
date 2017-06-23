package is.hail.methods

import breeze.linalg._
import breeze.numerics._
import com.sun.jna.Native
import is.hail.stats._
import is.hail.utils._

/**
  * Charlie Colley
  *   an implementation of the SKAT algorithm.
  * Input:
  *    G          (n x k) DenseMatrix
  *      The genotype matrix associated with n samples over k genetic variants
  *    Weights    (n x 1) DenseVector
  *      The weights associated with each of the genetic variants
  *      (TODO: Make this optional such that beta distributions or anonymous
  *      functions can be used)
  *    covs (n x m) DenseMatrix
  *      The covariates associated with each of the samples collected
  *      (ASSUMED to be continuous, not categorical data)
  *    Phenotypes (n x 1) DenseVector
  *      The phenotypes measured for each of the n samples in the population
  *      that we will regress on.
  */


class Skat(G:DenseMatrix[Double],weights:DenseVector[Double],
           covs:DenseMatrix[Double],phenotypes:DenseVector[Double]) {
  //initialize basic counters
  val n = covs.rows
  val m = covs.cols
  val k = G.cols

  //Declare variables
  var Q_factor: Option[DenseMatrix[Double]] = None
  var expectedMean:Option[DenseVector[Double]] = None
  var residual:Option[DenseVector[Double]] = None
  var parameterVector:Option[DenseVector[Double]] = None
  var variance:Option[Double] = None
  var VCS:Option[Double] = None


  // Phase 1: Solve least squares problem over the covariates
  //Returns (parameters,Q,mu,sigma)
  def regressCovariates():
  (DenseVector[Double], DenseMatrix[Double],DenseVector[Double],Double) = {
    //initialize basic counters

    //expand Covariates to include intercept
    var covariatesWithIntercept = DenseMatrix.ones[Double](n, m + 1)
    covariatesWithIntercept(::, 1 to m) := covs

    val QR = qr.reduced.impl_reduced_DM_Double(covariatesWithIntercept)
    parameterVector = Some(QR.r \ (QR.q.t * phenotypes))
    parameterVector match {
      case Some(p) =>
        expectedMean = Some(covariatesWithIntercept * p)
        residual = Some(phenotypes - expectedMean.get)
        expectedMean match {
          case Some(avg) =>

            variance = Some((residual.get dot residual.get) / (n - (m + 1)))
            Q_factor = Some(QR.q)
            (Q_factor, variance) match {
              case (Some(q), Some(s)) =>
                (p, q, avg, s)
              case _ => fatal("This can't happen")
            }
          case _ => fatal("This can't happen")
        }
      case _ => fatal("This can't happen")
    }
  }

  //Computes VCS by (y - mu)^TP^{1/2}GWG^TP{1/2}(y - mu)
  //Q,sigma is passed in to form P = sigma(I - QQ^T)
  //This method will have to change once building for a parallel paradigm
  def VarianceComponentStatistic():Double =
  {
    //Check if regression has run
    (Q_factor,expectedMean,variance,residual) match {
    case (Some(q),Some(mu),Some(sigma),Some(res)) =>
      //compute x = W^{1/2}G^T(y - mu)
      var x = G.t*res
      x = diag(sqrt(weights))*x

      //y^T y = Q
      VCS = Some(x.t * x)

      VCS match {
        case Some(v) => v
        case _ => fatal("This can't happen")
      }
    case _ => fatal("Regression hasn't run, call Skat" +
      ".regressCovariates() first")
    }


  }

  /*
  //Add in Davies Algorithm from his C code
  @native
  def qf(lb1:Array[Double],nc1:Array[Double],n1:Array[Double], r1:Int,
         sigma:Double, c1:Double,lim1:Int, acc:Double, trace:Array[Double],
         ifault:Array[Int])
  Native.register("ibs")
  */

  //Do we want to explicitly compute out the dense matrix first and pass it in?
  def pvalue():Double =
  {
    (Q_factor,variance) match {
      case (Some(q), Some(sigma)) =>
        val X = G * diag(sqrt(weights))
        val Y = sqrt(sigma) * (X - q * (q.t * X))
        //solve for the eigenvalues of the Gram Matrix for the chi-squared weights
        val gramMatrix = if (G.cols < G.rows) X * X.t else X.t * X
        val lambdas = eigSymD.justEigenvalues(gramMatrix)
        3.4
      case (None, None) =>
        fatal("VCS has not been computed run Skat.VarianceComponentStatistic" +
          "() first")
    }
  }
}


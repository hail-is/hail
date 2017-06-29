//Charlie Colley
//  29/6/2017
package is.hail.stats

import breeze.linalg._
import breeze.numerics._
import com.sun.jna.Native
import com.sun.jna.ptr.IntByReference
import is.hail.utils._

/** skat
  *   an implementation of the SKAT algorithm.
  * Input:
  *   G          (n x k) DenseMatrix
  *     The genotype matrix associated with n samples over k genetic variants
  *   covs       (n x m) DenseMatrix
  *     The covariates associated with each of the samples collected
  *     (ASSUMED to be continuous, not categorical data)
  *   weights    (n x 1) DenseVector
  *     The weights associated with each of the genetic variants
  *     (TODO: Make this optional such that beta distributions or anonymous
  *     functions can be used)
  *   phenotypes (n x 1) DenseVector
  *     The phenotypes measured for each of the n samples in the population
  *     that we will regress on.
  * Methods:
  *   fitCovariates
  *    - uses QR method for computing the least squares solution over the
  *      covariates with an added intercept.
  */


class skat(G: DenseMatrix[Double], covs: DenseMatrix[Double],
           phenotypes: DenseVector[Double], weights: DenseVector[Double]) {
  //initialize basic counters
  val n = covs.rows
  val m = covs.cols

  // Phase 1: Solve least squares problem over the covariates
  def fitCovariates():skatNullModel = {

    //expand Covariates to include intercept
    var covariatesWithIntercept = DenseMatrix.ones[Double](n, m + 1)
    covariatesWithIntercept(::, 1 to m) := covs

    val QR = qr.reduced.impl_reduced_DM_Double(covariatesWithIntercept)
    val beta = QR.r \ (QR.q.t * phenotypes)

    val expectedMean = covariatesWithIntercept * beta
    val residual = phenotypes - expectedMean
    val sigmaSqrd = (residual dot residual) / (n - (m + 1))

    new skatNullModel(QR.q, G * diag(sqrt(weights)), residual, sigmaSqrd)
    }
}


/**  skatNullModel
  *    a helper class produced used in the skat class. This class takes in a QR
  *    decomposition and residual from the regression algorithm run over the
  *    covariates and uses this to compute the skatStat (Q value from original
  *    SKAT paper a.k.a. variance component statistic) and the pvalue.
  *  Input:
  *    Q -           (n x n) matrix
  *      The Q from the QR factorization used to solve the linear least squares
  *      problem in skat.fitCovariates().
  *    weightedG -   (n x k) matrix
  *      The genotype matrix weighted by the sqrt of the weights passed in to
  *      the skat class.
  *    residual -    (n x 1) vector
  *      The residual computed from the least squares solution.
  *    sigmaSqrt -   Double
  *      The unbiased variance computed from the least squares solution.
  *   Methods:
  *     computeSkatStats
  *       - computes the variance component statistic and the associated pvalue
  *     computePvalue
  *       - a helper function to computeSkatStats which computes the pvalue of
  *         the skatStat.
  *     qf
  *       - a numerical integration routine for computing a value of the
  *         cumulative disitribution of a chi-squared distribution
  *         (Citation below).
  */
class skatNullModel(Q: DenseMatrix[Double],weightedG: DenseMatrix[Double],
                     residual: DenseVector[Double], sigmaSqrd: Double){

  /**   Davies Algorithm original C code
    *   Citation:
    *     Davies, Robert B. "The distribution of a linear combination of
    *     x2 random variables." Applied Statistics 29.3 (1980): 323-333.
    *   Software Link:
    *     http://www.robertnz.net/QF.htm
    */
   @native
  def qf(lb1: Array[Double], nc1: Array[Double], n1: Array[Int], r1: Int,
         sigma: Double, c1: Double, lim1: Int, acc: Double, trace:Array[Double],
         ifault: IntByReference): Double
  Native.register("ibs")


  def computePValue(skatStat: Double): Double =
  {

    //Compute coefficients of chi-squared distribution
    val TestMatrix =
      (weightedG.t * weightedG - weightedG.t * Q * Q.t * weightedG) * .5
    var lambdas =   eigSymD.justEigenvalues(TestMatrix).toArray
    val threshold = lambdas.sum / (lambdas.length * 1e5)
    lambdas = lambdas.filter(_ > threshold)

    //Initialize parameters to Davies' algorithm
    val terms = lambdas.length
    val noncentrality = Array.fill[Double](terms)(0.0)
    val degreesOfFreedom = Array.fill[Int](terms)(1)
    val trace = Array.fill[Double](7)(0.0)
    val fault = new IntByReference()
    val s = 0.0
    val accuracy = 1e-6
    val iterations = 10000
    val x = qf(lambdas,noncentrality,degreesOfFreedom,terms,s,skatStat,
               iterations,accuracy,trace,fault)


    fault.getValue match {
      case 0 => 1 - x
      case 1 => fatal(f"required accuracy of $accuracy NOT achieved")
      case 2 =>fatal("round-off error possibly significant")
      case 3 =>fatal("invalid parameters")
      case 4 =>fatal("unable to locate integration parameter")
      case 5 =>fatal("out of memory")
    }

  }

  //Computes the variance component statistic and the associate p value
  def computeSkatStats(): (Double, Double) =
  {
    //compute x = (y - mu)^T * G * W^{1/2}
    val x = residual.t * weightedG
    val skatStat = (x * x.t)/(2 * sigmaSqrd)

    (skatStat,computePValue(skatStat))

  }
}



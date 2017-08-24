package is.hail.stats

import breeze.linalg._
import is.hail.methods.SkatStat
import com.sun.jna.Native
import com.sun.jna.ptr.IntByReference


/** SkatModel
  *    A class which has all the algorithms for computing the p value of the
  *    Variance-component score statistic thats passed into the class.
  *  Input:
  *    qstat                   Double
  *      Variance-component score statistic at which to evaluate the cumulative
  *      distribution.
  *  Methods:
  *    computeLinearSkatStats
  *      computes the Skat stat for the linear case.
  *    computeLogisiticSkatStats
  *      computes the Skat stat for the logistic case.
  *    qfWrapper
  *      a function which interfaces with Davies' original C code for computing
  *      the p value.
  *    computePVal
  *      computes the eigendecomposition and plugs the eigenvalues in as the
  *      coefficients for the chi-square distribution to be numerically
  *      integrated to get the p value.
  */
class SkatModel(skatStat: Double) {

  /** Davies Algorithm original C code
    *   Citation:
    *     Davies, Robert B. "The distribution of a linear combination of
    *     x2 random variables." Applied Statistics 29.3 (1980): 323-333.
    *   Software Link:
    *     http://www.robertnz.net/QF.htm
    */
  @native
  def qfWrapper(lb1: Array[Double], nc1: Array[Double], n1: Array[Int], r1: Int,
    sigma: Double, c1: Double, lim1: Int, acc: Double, trace: Array[Double],
    ifault: IntByReference): Double

  Native.register("hail")

  /** computePVal
    *      The function which computes the p value of the Skat stat passed. This
    *    method computes the eigenvalues of a Gramian matrix which are the
    *    coefficients of the chi-square distribution describing the null-distribution
    *    for the Skat stat.
    * Input:
    *   Gramian: (m x m) DenseMatrix
    *     the matrix with the same eigenvalues of P0^{1/2}KP0^{1/2} from the original
    *     paper.
    * Returns:
    *   ((qstat, pval), fault)      SkatStat
    *     SKAT statistic and p-value computed by Davies Algorithm returned as a instance
    *     of the SkatStat class. With the statistics a integer indicating any issues
    *     running Davies Algorithm is also returned.
    */
  def computePVal(Gramian: DenseMatrix[Double]): SkatStat = {
    val allEvals = eigSymD.justEigenvalues(Gramian).toArray

    //compute threshold for eigenvalues
    var allEvalsSum = 0.0
    var i = 0
    while (i < allEvals.length) {
      allEvalsSum += allEvals(i)
      i += 1
    }
    val threshold = 1e-5 * allEvalsSum / allEvals.length

    //Initialize parameters to Davies' algorithm
    val evals = allEvals.filter(_ > threshold)
    val terms = evals.length

    val noncentrality = Array.ofDim[Double](terms)
    val dof = Array.fill[Int](terms)(1)
    val trace = Array.fill[Double](7)(0.0)
    val fault = new IntByReference()
    val s = 0.0
    val accuracy = 1e-6
    val iterations = 10000
    val x = qfWrapper(evals, noncentrality, dof, terms, s, skatStat, iterations, accuracy, trace, fault)

    SkatStat(skatStat, 1 - x, fault.getValue)

  }

  /** computeLinearSkatStats
    *   Computes the p value of the variance-component score statistic as
    *   defined by SKAT.
    * Input:
    *   GwGramian              (m x m) DenseMatrix
    *     The Gram matrix of the genotype matrix for n samples and m genetic
    *     variants with each column scaled by the corresponding variant weight.
    *   QtGwGramian            (m x m) DenseMatrix
    *     The Gram matrix of the projection of the weighted genotype matrix Gw
    *     by Q.t, where Q comes from the reduced QR decomposition of the
    *     covariate data.
    * Returns:
    *   SkatStat (see computePVal)
    */
  def computeLinearSkatStats(GwGramian: DenseMatrix[Double], QtGwGramian: DenseMatrix[Double]): SkatStat = {

    val variantGramian = 0.5 * (GwGramian - QtGwGramian)
    computePVal(variantGramian)
  }

  /** LogisticSKATModel
    *   Computes the p value of the variance-component score statistic as
    *   defined by SKAT from a logistic fit.
    * Input:
    *   X        (n x k) DenseMatrix
    *     The covariates matrix associated with the regression for n samples and
    *     k covariates.
    *   mu       (n x 1) DenseVector
    *     The expectation computed from the regression model.
    *   Gw       (n x m) DenseMatrix
    *     The weighted genotype matrix corresponding to the variants in a
    *     particular gene.
    * Returns:
    *   SkatStat (see computePVal)
    */
  def computeLogisticSkatStat(X: DenseMatrix[Double], mu: DenseVector[Double],
    Gw: DenseMatrix[Double]): SkatStat = {
      val V = diag(mu.map((x) => x * (1 - x)))
      val weightedX = V * X
      val P = V - weightedX * ((X.t * weightedX) \ weightedX.t)
      computePVal(Gw.t * P * Gw)
  }

}


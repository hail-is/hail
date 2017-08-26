package is.hail.stats

import breeze.linalg._
import is.hail.methods.SkatStat
import com.sun.jna.Native
import com.sun.jna.ptr.IntByReference

/** SkatModel
  *    A class which has all the algorithms for computing the p value of the
  *    Variance-component score statistic that's passed into the class.
  *  Input:
  *    qstat: Double
  *      Variance-component score statistic at which to evaluate the cumulative
  *      distribution.
  *  Methods:
  *    qfWrapper
  *      a function which interfaces with Davies' original C code for computing
  *      the p value.
  *    computePVal
  *      computes the eigendecomposition and plugs the eigenvalues in as the
  *      coefficients of the mixture of chi-square distribution to be numerically
  *      integrated to get the p-value.
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
    *    Function which computes the p-value of the SKAT statistic. This
    *      method computes the eigenvalues of a Gramian matrix which are the
    *      coefficients of the chi-square distribution describing the null-distribution
    *      for the SKAT statistic.
    * Input:
    *   gramian: (m x m) DenseMatrix
    *     Matrix (G * sqrt(W)).t * P_0 * G * sqrt(W) with the same non-zero eigenvalues as
    *     P_0^{1/2} * K * P_0^{1/2} from the original paper, where K = G * W * G.t
    * Returns:
    *   SkatStat:
    *     SKAT statistic, p-value computed by Davies Algorithm, and an integer indicating
    *     any return status of the Davies Algorithm.
    */
  def computePVal(gramian: DenseMatrix[Double]): SkatStat = {
    val allEvals = eigSymD.justEigenvalues(gramian).toArray

    //truncate eigenvalues at 1e-5 * mean
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
}


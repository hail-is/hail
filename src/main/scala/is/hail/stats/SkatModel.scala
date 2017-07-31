package is.hail.stats

import breeze.linalg._
import is.hail.methods.SkatStat
import com.sun.jna.Native
import com.sun.jna.ptr.IntByReference

/** SKATModel
  *   Computes the p value of the variance-component score statistic as 
  *   defined by SKAT.
  * Input:
  *   GwGrammian              (m x m) DenseMatrix
  *     The Gram matrix of the genotype matrix for n samples and m genetic
  *     variants with each column scaled by the corresponding variant weight.
  *   QtGwGrammian            (m x m) DenseMatrix
  *     The Gram matrix of the projection of the weighted genotype matrix Gw
  *     by Q.t, where Q comes from the reduced QR decomposition of the
  *     covariate data.
  *   qstat                   Double
  *     Variance-component score statistic at which to evaluate the cumulative
  *     distribution.
  * Returns:
  *   ((qstat, pval), fault)      (SkatStat, Int)
  *     SKAT statistic and p-value computed by Davies Algorithm returned as a instance
  *     of the SkatStat class. With the statistics a integer indicating any issues
  *     running Davies Algorithm is also returned.

object SkatModel {


  def synQf(lb1: Array[Double], nc1: Array[Double], n1: Array[Int], r1: Int,
    sigma: Double, c1: Double, lim1: Int, acc: Double, trace: Array[Double],
    ifault: IntByReference): Double = {
      this.synchronized{
        qfWrapper(lb1, nc1, n1, r1, sigma, c1, lim1, acc, trace, ifault)
  }
}
}
  */
class
SkatModel(GwGrammian: DenseMatrix[Double], QtGwGrammian: DenseMatrix[Double], skatStat: Double){

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

  def computeSkatStats(): (SkatStat,Int) = {

    val variantGrammian = 0.5 * (GwGrammian - QtGwGrammian)
    val allEvals = eigSymD.justEigenvalues(variantGrammian).toArray

    //compute threshold for eigenvalues
    var allEvalsSum = 0.0
    var i = 0
    while(i < allEvals.length){
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

    (SkatStat(skatStat, 1-x),fault.getValue)
  }
}

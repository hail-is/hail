package is.hail.stats

import breeze.linalg._
import is.hail.methods.SkatStat
import com.sun.jna.Native
import com.sun.jna.ptr.IntByReference

class SkatModel(q: Double) {

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
  
  // gramian is the m x m matrix (G * sqrt(W)).t * P_0 * (G * sqrt(W)) which has the same non-zero eigenvalues
  // as the n x n matrix in the paper P_0^{1/2} * (G * W * G.t) * P_0^{1/2}
  def computePVal(gramian: DenseMatrix[Double]): SkatStat = {
    val allEvals = eigSymD.justEigenvalues(gramian)

    // filter out those eigenvalues below the mean / 100k
    val threshold = 1e-5 * sum(allEvals) / allEvals.length
    val evals = allEvals.toArray.dropWhile(_ < threshold) // evals are increasing

    val terms = evals.length
    val noncentrality = Array.fill[Double](terms)(0.0)
    val dof = Array.fill[Int](terms)(1)
    val trace0 = Array.fill[Double](7)(0.0)
    val fault = new IntByReference()
    val s = 0.0
    val accuracy = 1e-8
    val iterations = 10000
    val x = qfWrapper(evals, noncentrality, dof, terms, s, q, iterations, accuracy, trace0, fault)
    val p = 1 - x
    
    SkatStat(q, p, fault.getValue)
  }
}

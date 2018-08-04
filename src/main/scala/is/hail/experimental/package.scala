package is.hail

import is.hail.stats._

package object experimental {

  def findMaxAC(af: Double, an: Int, ci: Double = .95): Int = {
   if (af == 0)
      0
    else {
      val quantile_limit = ci // ci for one-sided, 1-(1-ci)/2 for two-sided
      val max_ac = qpois(quantile_limit, an * af)
      max_ac
    }
  }

  def calcFilterAlleleFreq(ac: Int, an: Int, ci: Double = .95, lower: Double = 1e-10, upper: Double = 2, tol: Double = 1e-7, precision: Double = 1e-6): Double = {
    if (ac <= 0 || an == 0)
      0.0
    else {
      var f = (af: Double) => ac.toDouble - 1 - qpois(ci, an.toDouble * af)
      val root = uniroot(f, lower, upper, tol)
      val rounder = 1d / (precision / 100d)
      var max_af = math.round(root.getOrElse(0.0) * rounder) / rounder
      while (findMaxAC(max_af, an) < ac) {
        max_af += precision
      }
      max_af - precision
    }
  }

  def calcFilterAlleleFreq(ac: Int, an: Int, ci: Double): Double = calcFilterAlleleFreq(ac, an, ci, lower = 1e-10, upper = 2, tol = 1e-7, precision = 1e-6)

}

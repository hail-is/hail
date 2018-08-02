package is.hail.experimental

import is.hail.expr.types._

object ExperimentalFunctions extends RegistryFunctions {

  def registerAll() {
    registerScalaFunction("filtering_allele_frequency", TInt32(), TInt32(), TFloat64(), TFloat64())(getClass, "calcFilterAlleleFreq")

  }

  def calcFilterAlleleFreq(ac: Int, an: Int, ci: Double): Double = calcFilterAlleleFreq(ac, an, ci, lower = 1e-10, upper = 2, tol = 1e-7, precision = 1e-6)

}

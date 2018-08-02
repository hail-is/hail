package is.hail.experimental

import is.hail.expr.types._

object ExperimentalFunctions extends RegistryFunctions {

  def registerAll() {
    val experimentalPackageClass = Class.forName("is.hail.experimental.package$")

    registerScalaFunction("filtering_allele_frequency", TInt32(), TInt32(), TFloat64(), TFloat64())(experimentalClass, "calcFilterAlleleFreq")

  }

}

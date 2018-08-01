package is.hail.expr.ir.functions

import is.hail.expr.types._

object ExperimentalFunctions extends RegistryFunctions {

  def registerAll() {
    val experimentalPackageClass = Class.forName("is.hail.experimental.package$")

    registerScalaFunction("faf", TInt32(), TInt32(), TFloat64(), TFloat64())(experimentalPackageClass, "calcFreqFilter")

  }

}

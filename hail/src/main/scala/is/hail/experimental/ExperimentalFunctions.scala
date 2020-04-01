package is.hail.experimental

import is.hail.expr.ir.functions._
import is.hail.expr.types.physical.{PCanonicalArray, PFloat64, PType}
import is.hail.expr.types.virtual.{TArray, TFloat64, TInt32, Type}

object ExperimentalFunctions extends RegistryFunctions {

  def registerAll() {
    val experimentalPackageClass = Class.forName("is.hail.experimental.package$")

    registerScalaFunction("filtering_allele_frequency", Array(TInt32, TInt32, TFloat64), TFloat64, (_: Type, pt: Seq[PType]) => PFloat64())(experimentalPackageClass, "calcFilterAlleleFreq")
    registerWrappedScalaFunction("haplotype_freq_em", TArray(TInt32), TArray(TFloat64), (_: Type, pt: PType) => PCanonicalArray(PFloat64(true)))(experimentalPackageClass, "haplotypeFreqEM")
  }
}
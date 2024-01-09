package is.hail.experimental

import is.hail.expr.ir.functions._
import is.hail.types.physical.{PCanonicalArray, PFloat64, PType}
import is.hail.types.physical.stypes.SType
import is.hail.types.physical.stypes.concrete.SIndexablePointer
import is.hail.types.virtual.{TArray, TFloat64, TInt32, Type}

object ExperimentalFunctions extends RegistryFunctions {

  def registerAll() {
    val experimentalPackageClass = Class.forName("is.hail.experimental.package$")

    registerScalaFunction(
      "filtering_allele_frequency",
      Array(TInt32, TInt32, TFloat64),
      TFloat64,
      null,
    )(experimentalPackageClass, "calcFilterAlleleFreq")
    registerWrappedScalaFunction1(
      "haplotype_freq_em",
      TArray(TInt32),
      TArray(TFloat64),
      (_: Type, pt: SType) => SIndexablePointer(PCanonicalArray(PFloat64(true))),
    )(experimentalPackageClass, "haplotypeFreqEM")
  }
}

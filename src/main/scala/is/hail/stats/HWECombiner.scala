package is.hail.stats

import is.hail.annotations.Annotation
import is.hail.expr.{Field, TDouble, TStruct}
import is.hail.utils._
import is.hail.variant.Genotype

object HWECombiner {
  def signature = TStruct(Array(
    ("rExpectedHetFrequency", TDouble, "Expected rHeterozygosity based on Hardy Weinberg Equilibrium"),
    ("pHWE", TDouble, "p-value")
  ).zipWithIndex.map { case ((n, t, d), i) => Field(n, t, i, Map(("desc", d))) })
}

class HWECombiner extends Serializable {
  var nHomRef = 0
  var nHet = 0
  var nHomVar = 0

  def merge(gt:Genotype): HWECombiner = {
    if (Genotype.isHomRef(gt))
      nHomRef += 1
    else if (Genotype.isHet(gt))
      nHet += 1
    else if (Genotype.isHomVar(gt))
      nHomVar += 1

    this
  }

  def merge(other: HWECombiner): HWECombiner = {
    nHomRef += other.nHomRef
    nHet += other.nHet
    nHomVar += other.nHomVar

    this
  }

  def n = nHomRef + nHet + nHomVar
  def nA = nHet + 2 * nHomRef.min(nHomVar)

  def lh = LeveneHaldane(n, nA)

  def asAnnotation: Annotation = Annotation(divOption(lh.getNumericalMean, n).orNull, lh.exactMidP(nHet))
}

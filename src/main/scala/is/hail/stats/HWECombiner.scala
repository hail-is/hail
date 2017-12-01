package is.hail.stats

import is.hail.annotations.Annotation
import is.hail.expr.{Field, TFloat64, TStruct}
import is.hail.utils._
import is.hail.variant.Call

object HWECombiner {
  def signature = TStruct("rExpectedHetFrequency" -> TFloat64(), "pHWE" -> TFloat64())
}

class HWECombiner extends Serializable {
  var nHomRef = 0
  var nHet = 0
  var nHomVar = 0

  def merge(gt: Call): HWECombiner = {
    if (gt!= null) {
      if (Call.isHomRef(gt))
        nHomRef += 1
      else if (Call.isHet(gt))
        nHet += 1
      else if (Call.isHomVar(gt))
        nHomVar += 1
    }

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

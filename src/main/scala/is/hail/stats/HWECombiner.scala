package is.hail.stats

import is.hail.annotations.Annotation
import is.hail.expr.types._
import is.hail.utils._
import is.hail.variant.Call

import scala.annotation.switch

object HWECombiner {
  def signature = TStruct("rExpectedHetFrequency" -> TFloat64(), "pHWE" -> TFloat64())
}

class HWECombiner extends Serializable {
  var nHomRef = 0
  var nHet = 0
  var nHomVar = 0

  def merge(c: Call): HWECombiner = {
    (Call.ploidy(c): @switch) match {
      case 2 =>
        if (Call.isHomRef(c))
          nHomRef += 1
        else if (Call.isHet(c))
          nHet += 1
        else if (Call.isHomVar(c))
          nHomVar += 1
      case _ =>
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

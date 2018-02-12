package is.hail.stats

import is.hail.annotations.Annotation
import is.hail.expr.types._
import is.hail.utils._
import is.hail.variant.{Call, Genotype}

object InbreedingCombiner {
  def signature = TStruct(
    "f_stat" -> TFloat64(),
    "n_called" -> TInt64(),
    "expected_homs" -> TFloat64(),
    "observed_homs" -> TInt64())
}

class InbreedingCombiner extends Serializable {
  var nCalled = 0L
  var expectedHoms = 0d
  var observedHoms = 0L

  def merge(c: Call, af: Double): InbreedingCombiner = {
    nCalled += 1
    expectedHoms += 1 - (2 * af * (1 - af))

    if (Call.isHomRef(c) || Call.isHomVar(c))
      observedHoms += 1

    this
  }

  def merge(other: InbreedingCombiner): InbreedingCombiner = {
    nCalled += other.nCalled
    expectedHoms += other.expectedHoms
    observedHoms += other.observedHoms
    this
  }

  def Fstat: Option[Double] = divOption(observedHoms - expectedHoms, nCalled - expectedHoms)

  def asAnnotation: Annotation = Annotation(Fstat.orNull, nCalled, expectedHoms, observedHoms)
}

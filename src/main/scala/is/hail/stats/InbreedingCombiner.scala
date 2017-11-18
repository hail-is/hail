package is.hail.stats


import is.hail.annotations.Annotation
import is.hail.expr.{Field, TFloat64, TInt64, TStruct}
import is.hail.utils._
import is.hail.variant.{Call, Genotype}

object InbreedingCombiner {
  def signature = TStruct(
    "Fstat" -> TFloat64(),
    "nTotal" -> TInt64(),
    "nCalled" -> TInt64(),
    "expectedHoms" -> TFloat64(),
    "observedHoms" -> TInt64())
}

class InbreedingCombiner extends Serializable {
  var nCalled = 0L
  var expectedHoms = 0d
  var observedHoms = 0L
  var total = 0L

  def merge(gt: Call, af: Double): InbreedingCombiner = {
    total += 1
    if (gt != null) {
      nCalled += 1
      expectedHoms += 1 - (2 * af * (1 - af))

      if (Call.isHomRef(gt) || Call.isHomVar(gt))
        observedHoms += 1
    }
    this
  }

  def merge(other: InbreedingCombiner): InbreedingCombiner = {
    nCalled += other.nCalled
    expectedHoms += other.expectedHoms
    observedHoms += other.observedHoms
    total += other.total
    this
  }

  def Fstat: Option[Double] = divOption(observedHoms - expectedHoms, nCalled - expectedHoms)

  def asAnnotation: Annotation = Annotation(Fstat.orNull, total, nCalled, expectedHoms, observedHoms)
}

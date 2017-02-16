package is.hail.stats

import is.hail.annotations.Annotation
import is.hail.expr.{TDouble, TInt, TStruct}
import is.hail.utils._
import is.hail.variant.Genotype

object InbreedingCombiner {
  def signature = TStruct("Fstat" -> TDouble,
    "nTotal" -> TInt,
    "nCalled" -> TInt,
    "expectedHoms" -> TDouble,
    "observedHoms" -> TInt)
}

class InbreedingCombiner extends Serializable {
  var nCalled = 0L
  var expectedHoms = 0d
  var observedHoms = 0L
  var total = 0L

  def merge(gt:Genotype, af: Double): InbreedingCombiner = {
    total += 1
    if (gt.isCalled) {
      nCalled += 1
      expectedHoms += 1 - (2 * af * (1 - af))

      if (gt.isHomRef || gt.isHomVar)
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

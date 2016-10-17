package org.broadinstitute.hail.stats

import org.broadinstitute.hail.annotations.{Annotation, _}
import org.broadinstitute.hail.expr.{TDouble, TInt, TStruct}
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant.Genotype

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

  def merge(gt:Genotype, maf: Double): InbreedingCombiner = {
    total += 1
    if (gt.isCalled) {
      nCalled += 1
      expectedHoms += 1 - (2 * maf * (1 - maf))

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

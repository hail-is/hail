package org.broadinstitute.hail.methods

import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.variant.Genotype
import org.broadinstitute.hail.utils._

object InbreedingCombiner {
  def signature = TStruct("Fstat" -> TDouble,
    "nTotal" -> TInt,
    "nCalled" -> TInt,
    "expectedHoms" -> TDouble,
    "observedHoms" -> TInt)
}

class InbreedingCombiner extends Serializable {
  var nCalled = 0
  var expectedHoms = 0d
  var observedHoms = 0
  var total = 0

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
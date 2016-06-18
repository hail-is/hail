package org.broadinstitute.hail.methods

import org.broadinstitute.hail.variant.Genotype
import org.broadinstitute.hail.Utils._

class InbreedingCombiner extends Serializable {
  var nCalled = 0
  var expectedHoms = 0d
  var observedHoms = 0
  var total = 0

  def addCount(gt:Genotype, maf: Double): InbreedingCombiner = {
    total += 1
    if (gt.isCalled) {
      nCalled += 1
      expectedHoms += 1 - (2 * maf * (1 - maf))

      if (gt.isHomRef || gt.isHomVar)
        observedHoms += 1
    }
    this
  }

  def combineCounts(other: InbreedingCombiner): InbreedingCombiner = {
    nCalled += other.nCalled
    expectedHoms += other.expectedHoms
    observedHoms += other.observedHoms
    total += other.total
    this
  }

  def Fstat: Option[Double] = divOption(observedHoms - expectedHoms, nCalled - expectedHoms)
}
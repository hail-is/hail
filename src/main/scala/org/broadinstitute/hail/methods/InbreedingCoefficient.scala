package org.broadinstitute.hail.methods

import org.broadinstitute.hail.variant.Genotype
import org.broadinstitute.hail.Utils._

class InbreedingCombiner extends Serializable {
  var N = 0.0
  var E = 0.0
  var O = 0.0
  var T = 0.0

  def addCount(gt:Genotype, maf: Double): InbreedingCombiner = {
    T += 1
    if (gt.isCalled) {
      N += 1
      E += 1.0 - (2.0*maf*(1.0-maf))

      if (gt.isHomRef || gt.isHomVar)
        O += 1
    }
    this
  }

  def combineCounts(other: InbreedingCombiner): InbreedingCombiner = {
    N += other.N
    E += other.E
    O += other.O
    T += other.T
    this
  }

  def F: Option[Double] = divOption(O - E, N - E)
}
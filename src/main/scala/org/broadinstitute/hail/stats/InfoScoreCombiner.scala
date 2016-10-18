package org.broadinstitute.hail.stats

import org.broadinstitute.hail.annotations.{Annotation, _}
import org.broadinstitute.hail.expr.{TDouble, TInt, TStruct}
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant.Genotype

object InfoScoreCombiner {
  def signature = TStruct("score" -> TDouble, "nIncluded" -> TInt)
}

class InfoScoreCombiner extends Serializable {
  var result = 0d
  var expectedAlleleCount = 0d
  var totalDosage = 0d
  var nIncluded = 0

  def expectedVariance(dosage: Array[Double], mean: Double): Double = (dosage(1) + 4 * dosage(2)) - (mean * mean)

  def merge(g: Genotype): InfoScoreCombiner = {
    g.dosage.foreach { dx =>
      val mean = dx(1) + 2 * dx(2)
      result += expectedVariance(dx, mean)
      expectedAlleleCount += mean
      totalDosage += dx.sum
      nIncluded += 1
    }
    this
  }

  def merge(that: InfoScoreCombiner): InfoScoreCombiner = {
    result += that.result
    expectedAlleleCount += that.expectedAlleleCount
    totalDosage += that.totalDosage
    nIncluded += that.nIncluded
    this
  }

  def thetaMLE = divOption(expectedAlleleCount, totalDosage)

  def imputeInfoScore(theta: Double) =
    if (theta == 1.0 || theta == 0.0)
      Some(1d)
    else if (nIncluded == 0)
      None
    else
      Some(1d - ((result / nIncluded) / (2 * theta * (1 - theta))))

  def asAnnotation: Annotation = {
    val score = thetaMLE.flatMap { theta => imputeInfoScore(theta) }

    Annotation(score.orNull, nIncluded)
  }
}
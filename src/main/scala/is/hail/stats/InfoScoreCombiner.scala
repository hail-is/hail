package is.hail.stats

import is.hail.annotations.Annotation
import is.hail.expr.{Field, TDouble, TInt, TStruct}
import is.hail.utils._
import is.hail.variant.Genotype

object InfoScoreCombiner {
  def signature = TStruct(Array(
    ("score", TDouble, "IMPUTE info score"),
    ("nIncluded", TInt, "Number of samples with non-missing genotype probability distribution")
  ).zipWithIndex.map { case ((n, t, d), i) => Field(n, t, i, Map(("desc", d))) })
}

class InfoScoreCombiner extends Serializable {
  var result = 0d
  var expectedAlleleCount = 0d
  var totalDosage = 0d
  var nIncluded = 0

  def expectedVariance(gp: Array[Double], mean: Double): Double = (gp(1) + 4 * gp(2)) - (mean * mean)

  def merge(g: Genotype): InfoScoreCombiner = {
    g.gp.foreach { dx =>
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
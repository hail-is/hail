package is.hail.stats

import is.hail.annotations.{Annotation, Region, RegionValueBuilder}
import is.hail.expr.types._
import is.hail.utils._

object InfoScoreCombiner {
  def signature = TStruct("score" -> TFloat64(), "n_included" -> TInt32())
}

class InfoScoreCombiner extends Serializable {
  var result = 0d
  var expectedAlleleCount = 0d
  var totalDosage = 0d
  var nIncluded = 0
  val typ = TArray(TFloat64())

  def expectedVariance(gp: IndexedSeq[java.lang.Double], mean: Double): Double = (gp(1) + 4 * gp(2)) - (mean * mean)

  def merge(gp: IndexedSeq[java.lang.Double]): InfoScoreCombiner = {
    if (gp != null) {
      if (gp.length != 3)
        fatal(s"'info_score': invalid GP array: expected 3 elements, found ${gp.length}: [${gp.mkString(", ")}]")
      if (gp.contains(null))
        fatal(s"'info_score': invalid GP array: missing elements are not supported: [${gp.mkString(", ")}]")
      val mean = gp(1) + 2 * gp(2)
      result += expectedVariance(gp, mean)
      expectedAlleleCount += mean
      totalDosage += gp(0)
      totalDosage += gp(1)
      totalDosage += gp(2)
      nIncluded += 1
    }
    this
  }

  def merge(region: Region, offset: Long): InfoScoreCombiner = {
    val length = typ.loadLength(region, offset)
    if (length != 3)
      fatal(s"'info_score': invalid GP array: expected 3 elements, found $length.")

    var i = 0
    var mean = 0d
    var variance = 0d

    while (i < 3) {
      if (!typ.isElementDefined(region, offset, i))
        fatal(s"'info_score': invalid GP array: missing elements are not supported.")
      val gpi = region.loadDouble(typ.loadElement(region, offset, i))
      totalDosage += gpi
      mean += gpi * i
      variance += gpi * i * i
      i += 1
    }

    variance -= mean * mean
    nIncluded += 1
    result += variance
    expectedAlleleCount += mean

    this
  }

  def merge(that: InfoScoreCombiner): InfoScoreCombiner = {
    result += that.result
    expectedAlleleCount += that.expectedAlleleCount
    totalDosage += that.totalDosage
    nIncluded += that.nIncluded
    this
  }

  def thetaMLE: Option[Double] = divOption(expectedAlleleCount, totalDosage)

  def imputeInfoScore(theta: Double): Option[Double] =
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

  def result(rvb: RegionValueBuilder) {
    rvb.startStruct()

    val theta = divNull(expectedAlleleCount, totalDosage)
    if (theta == null)
      rvb.setMissing()
    else {
      if (theta == 1.0 || theta == 0.0)
        rvb.addDouble(1.0)
      else if (nIncluded == 0)
        rvb.setMissing()
      else
        rvb.addDouble(1d - ((result / nIncluded) / (2 * theta * (1 - theta))))
    }

    rvb.addInt(nIncluded)

    rvb.endStruct()
  }

  def clear() {
    result = 0d
    expectedAlleleCount = 0d
    totalDosage = 0d
    nIncluded = 0
  }

  def copy(): InfoScoreCombiner = {
    val c = new InfoScoreCombiner()
    c.result = result
    c.expectedAlleleCount = expectedAlleleCount
    c.totalDosage = totalDosage
    c.nIncluded = nIncluded
    c
  }
}

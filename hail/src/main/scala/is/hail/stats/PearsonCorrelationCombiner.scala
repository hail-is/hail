package is.hail.stats

import is.hail.annotations.{Annotation, RegionValueBuilder}

class PearsonCorrelationCombiner extends Serializable {
  // formula from https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

  var xSum: Double = 0d
  var ySum: Double = 0d
  var xSqSum: Double = 0d
  var ySqSum: Double = 0d
  var xySum: Double = 0d
  var n: Long = 0L

  def merge(x: Double, y: Double) {
    xSum += x
    ySum += y
    xSqSum += x * x
    ySqSum += y * y
    xySum += x * y
    n += 1
  }

  def merge(other: PearsonCorrelationCombiner) {
    xSum += other.xSum
    ySum += other.ySum
    xSqSum += other.xSqSum
    ySqSum += other.ySqSum
    xySum += other.xySum
    n += other.n
  }

  private def computeResult(): Double = (n * xySum - xSum * ySum) / math.sqrt((n * xSqSum - xSum * xSum) * (n * ySqSum - ySum * ySum))

  def result(): Annotation = if (n > 0) computeResult() else null

  def result(rvb: RegionValueBuilder){
    if (n > 0) {
      val corr = computeResult()
      rvb.addDouble(corr)
    } else rvb.setMissing()
  }

  def copy(): PearsonCorrelationCombiner = {
    val newComb = new PearsonCorrelationCombiner()
    newComb.xSum = xSum
    newComb.ySum = ySum
    newComb.xSqSum = xSqSum
    newComb.ySqSum = ySqSum
    newComb.xySum = xySum
    newComb.n = n
    newComb
  }

  def clear() {
    xSum = 0d
    ySum = 0d
    xSqSum = 0d
    ySqSum = 0d
    xySum = 0d
    n = 0L
  }
}

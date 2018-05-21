package is.hail.annotations.aggregators

import is.hail.annotations._

class RegionValueSumLongAggregator extends RegionValueAggregator {
  private var sum: Long = 0L

  def seqOp(region: Region, l: Long, missing: Boolean) {
    if (!missing)
      sum += l
  }

  def combOp(agg2: RegionValueAggregator) {
    sum += agg2.asInstanceOf[RegionValueSumLongAggregator].sum
  }

  def result(rvb: RegionValueBuilder) {
    rvb.addLong(sum)
  }

  def copy(): RegionValueSumLongAggregator = new RegionValueSumLongAggregator()

  def clear() {
    sum = 0L
  }
}

class RegionValueSumDoubleAggregator extends RegionValueAggregator {
  private var sum: Double = 0.0

  def seqOp(region: Region, d: Double, missing: Boolean) {
    if (!missing)
      sum += d
  }

  def combOp(agg2: RegionValueAggregator) {
    sum += agg2.asInstanceOf[RegionValueSumDoubleAggregator].sum
  }

  def result(rvb: RegionValueBuilder) {
    rvb.addDouble(sum)
  }

  def copy(): RegionValueSumDoubleAggregator = new RegionValueSumDoubleAggregator()

  def clear() {
    sum = 0.0
  }
}

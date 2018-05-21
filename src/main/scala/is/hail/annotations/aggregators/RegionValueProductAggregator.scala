package is.hail.annotations.aggregators

import is.hail.annotations._

class RegionValueProductLongAggregator extends RegionValueAggregator {
  private var product: Long = 1L

  def seqOp(region: Region, l: Long, missing: Boolean) {
    if (!missing)
      product *= l
  }

  def combOp(agg2: RegionValueAggregator) {
    product *= agg2.asInstanceOf[RegionValueProductLongAggregator].product
  }

  def result(rvb: RegionValueBuilder) {
    rvb.addLong(product)
  }

  def copy(): RegionValueProductLongAggregator = new RegionValueProductLongAggregator()

  def clear() {
    product = 1L
  }
}

class RegionValueProductDoubleAggregator extends RegionValueAggregator {
  private var product: Double = 1.0

  def seqOp(region: Region, d: Double, missing: Boolean) {
    if (!missing)
      product *= d
  }

  def combOp(agg2: RegionValueAggregator) {
    product *= agg2.asInstanceOf[RegionValueProductDoubleAggregator].product
  }

  def result(rvb: RegionValueBuilder) {
    rvb.addDouble(product)
  }

  def copy(): RegionValueProductDoubleAggregator = new RegionValueProductDoubleAggregator()

  def clear() {
    product = 1.0
  }
}

package is.hail.annotations.aggregators

import is.hail.annotations.RegionValueBuilder

class RegionValueCountAggregator extends RegionValueAggregator {
  private var count: Long = 0

  def seqOp(dummy: Int, missing: Boolean) {
    count += 1
  }

  def combOp(agg2: RegionValueAggregator) {
    count += agg2.asInstanceOf[RegionValueCountAggregator].count
  }

  def result(rvb: RegionValueBuilder) {
    rvb.addLong(count)
  }

  def copy(): RegionValueCountAggregator = new RegionValueCountAggregator()
}

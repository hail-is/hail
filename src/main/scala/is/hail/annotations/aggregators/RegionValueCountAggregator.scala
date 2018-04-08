package is.hail.annotations.aggregators

import is.hail.annotations.{Region, RegionValueBuilder}

class RegionValueCountAggregator extends RegionValueAggregator {
  private var count: Long = 0

  def seqOp(offset: Long, missing: Boolean) {
    count += 1
  }

  def seqOp(region: Region, off: Long, missing: Boolean) {
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

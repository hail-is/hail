package is.hail.annotations.aggregators

import is.hail.annotations._

trait RegionValueAggregator extends Serializable {
  def seqOp(region: Region, off: Long, missing: Boolean): Unit

  def combOp(agg2: RegionValueAggregator): Unit

  def result(rvb: RegionValueBuilder)

  def copy(): RegionValueAggregator
}

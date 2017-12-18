package is.hail.annotations.aggregators

import is.hail.annotations._
import is.hail.expr.RegionValueAggregator

class RegionValueFractionAggregator extends RegionValueAggregator {
  private var trues = 0L
  private var total = 0L

  def seqOp(i: Boolean, missing: Boolean) {
    total += 1
    if (!missing && i)
      trues += 1
  }

  def seqOp(region: Region, off: Long, missing: Boolean) {
    seqOp(region.loadBoolean(off), missing)
  }

  def result(region: Region): Long =
    // FIXME: divide by zero should be NA not NaN
    region.appendDouble(trues.toDouble / total)

  def combOp(agg2: RegionValueAggregator) {
    val other = agg2.asInstanceOf[RegionValueFractionAggregator]
    trues += other.trues
    total += other.total
  }

  def copy() = new RegionValueFractionAggregator()
}

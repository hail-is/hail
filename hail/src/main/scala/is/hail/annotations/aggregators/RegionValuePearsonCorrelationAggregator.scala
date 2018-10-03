package is.hail.annotations.aggregators

import is.hail.annotations.{Region, RegionValueBuilder}
import is.hail.stats.PearsonCorrelationCombiner

class RegionValuePearsonCorrelationAggregator extends RegionValueAggregator {
  var combiner = new PearsonCorrelationCombiner()

  def seqOp(region: Region, x: Double, xm: Boolean, y: Double, ym: Boolean) {
    if (!xm && !xm)
      combiner.merge(x, y)
  }

  def combOp(agg2: RegionValueAggregator) {
    val other = agg2.asInstanceOf[RegionValuePearsonCorrelationAggregator]
    combiner.merge(other.combiner)
  }

  def result(rvb: RegionValueBuilder) {
    combiner.result(rvb)
  }

  def newInstance(): RegionValuePearsonCorrelationAggregator = new RegionValuePearsonCorrelationAggregator()

  def copy(): RegionValuePearsonCorrelationAggregator = {
    val rva = new RegionValuePearsonCorrelationAggregator()
    rva.combiner = combiner.copy()
    rva
  }

  def clear() {
    combiner.clear()
  }
}

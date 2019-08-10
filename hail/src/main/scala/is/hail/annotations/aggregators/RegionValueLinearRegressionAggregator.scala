package is.hail.annotations.aggregators

import is.hail.annotations.{Region, RegionValueBuilder}
import is.hail.expr.types.virtual.Type
import is.hail.stats.LinearRegressionCombiner

object RegionValueLinearRegressionAggregator {
  def typ = LinearRegressionCombiner.typ
}

class RegionValueLinearRegressionAggregator(k: Int, k0: Int, xType: Type) extends RegionValueAggregator {
  var combiner = new LinearRegressionCombiner(k, k0, xType.physicalType)

  def seqOp(region: Region, y: Double, ym: Boolean, xsOffset: Long, xsMissing: Boolean) {
    if (!ym && !xsMissing) {
      combiner.merge(region, y, xsOffset)
    }
  }

  def combOp(agg2: RegionValueAggregator) {
    val other = agg2.asInstanceOf[RegionValueLinearRegressionAggregator]
    combiner.merge(other.combiner)
  }

  def result(rvb: RegionValueBuilder) {
    combiner.result(rvb)
  }

  def newInstance(): RegionValueLinearRegressionAggregator = new RegionValueLinearRegressionAggregator(k, k0, xType)

  def copy(): RegionValueLinearRegressionAggregator = {
    val rva = new RegionValueLinearRegressionAggregator(k, k0, xType)
    rva.combiner = combiner.copy()
    rva
  }

  def clear() {
    combiner.clear()
  }
}

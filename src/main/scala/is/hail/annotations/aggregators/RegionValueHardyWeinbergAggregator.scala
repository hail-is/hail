package is.hail.annotations.aggregators

import is.hail.annotations.{Region, RegionValueBuilder}
import is.hail.expr.types.TStruct
import is.hail.stats.HWECombiner
import is.hail.variant.Call

object RegionValueHardyWeinbergAggregator {
  def typ: TStruct = HWECombiner.signature
}

class RegionValueHardyWeinbergAggregator extends RegionValueAggregator {
  private var combiner = new HWECombiner()

  def seqOp(region: Region, c: Call, missing: Boolean) {
    if (!missing)
      combiner.merge(c)
  }

  override def combOp(agg2: RegionValueAggregator) {
    combiner.merge(agg2.asInstanceOf[RegionValueHardyWeinbergAggregator].combiner)
  }

  override def result(rvb: RegionValueBuilder) {
    combiner.result(rvb)
  }

  override def copy(): RegionValueAggregator = new RegionValueHardyWeinbergAggregator()

  override def clear() {
    combiner = new HWECombiner()
  }
}

package is.hail.annotations.aggregators

import is.hail.annotations.{Region, RegionValueBuilder}
import is.hail.expr.types.TStruct
import is.hail.stats.{CallStats, CallStatsCombiner}
import is.hail.variant.Call

object RegionValueCallStatsAggregator {
  val typ: TStruct = CallStats.schema
}

class RegionValueCallStatsAggregator extends RegionValueAggregator {
  private var combiner: CallStatsCombiner = _

  def initOp(nAlleles: Int, missing: Boolean): Unit = {
    if (!missing)
      combiner = new CallStatsCombiner(nAlleles)
  }

  def seqOp(region: Region, x: Call, missing: Boolean): Unit = {
    if (combiner != null && !missing)
      combiner.merge(x)
  }

  def combOp(agg2: RegionValueAggregator): Unit = {
    val other = agg2.asInstanceOf[RegionValueCallStatsAggregator]
    if (other.combiner != null) {
      if (combiner == null)
        combiner = new CallStatsCombiner(other.combiner.nAlleles)
      combiner.merge(other.combiner)
    }
  }

  def result(rvb: RegionValueBuilder): Unit = {
    if (combiner != null) {
      combiner.result(rvb)
    } else
      rvb.setMissing()
  }

  def newInstance(): RegionValueCallStatsAggregator = new RegionValueCallStatsAggregator()

  def copy() = {
    val rva = new RegionValueCallStatsAggregator()
    rva.combiner = combiner.copy()
    rva
  }

  def clear() {
    combiner = null
  }
}

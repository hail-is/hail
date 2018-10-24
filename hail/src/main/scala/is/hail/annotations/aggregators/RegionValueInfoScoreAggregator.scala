package is.hail.annotations.aggregators

import is.hail.annotations.{Region, RegionValueBuilder}
import is.hail.expr.types.Type
import is.hail.expr.types.physical.PType
import is.hail.stats.InfoScoreCombiner

object RegionValueInfoScoreAggregator {
  def typ = InfoScoreCombiner.signature
}

class RegionValueInfoScoreAggregator(typ: PType) extends RegionValueAggregator {
  var combiner = new InfoScoreCombiner(typ)

  def seqOp(region: Region, offset: Long, missing: Boolean) {
    if (!missing) {
      combiner.merge(region, offset)
    }
  }

  override def combOp(agg2: RegionValueAggregator) {
    combiner.merge(agg2.asInstanceOf[RegionValueInfoScoreAggregator].combiner)
  }

  override def result(rvb: RegionValueBuilder) {
    combiner.result(rvb)
  }

  override def newInstance(): RegionValueAggregator = new RegionValueInfoScoreAggregator(typ)

  override def copy(): RegionValueInfoScoreAggregator = {
    val rva = new RegionValueInfoScoreAggregator(typ)
    rva.combiner = combiner.copy()
    rva
  }

  override def clear() {
    combiner.clear()
  }
}

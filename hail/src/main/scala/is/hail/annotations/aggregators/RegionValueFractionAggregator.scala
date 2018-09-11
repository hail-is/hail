package is.hail.annotations.aggregators

import is.hail.annotations._

class RegionValueFractionAggregator extends RegionValueAggregator {
  private var trues = 0L
  private var total = 0L

  def seqOp(region: Region, i: Boolean, missing: Boolean) {
    total += 1
    if (!missing && i)
      trues += 1
  }

  def result(rvb: RegionValueBuilder) {
    if (total == 0)
      rvb.setMissing()
    else
      rvb.addDouble(trues.toDouble / total)
  }

  def combOp(agg2: RegionValueAggregator) {
    val other = agg2.asInstanceOf[RegionValueFractionAggregator]
    trues += other.trues
    total += other.total
  }

  def newInstance() = new RegionValueFractionAggregator()

  override def copy(): RegionValueFractionAggregator = {
    val rva = new RegionValueFractionAggregator()
    rva.trues = trues
    rva.total = total
    rva
  }

  def clear() {
    trues = 0L
    total = 0L
  }
}

package is.hail.annotations.aggregators

import is.hail.annotations.{Region, RegionValueBuilder}
import is.hail.stats.InbreedingCombiner
import is.hail.variant.Call

object RegionValueInbreedingAggregator {
  def typ = InbreedingCombiner.signature
}

class RegionValueInbreedingAggregator extends RegionValueAggregator {
  var combiner: InbreedingCombiner = new InbreedingCombiner()

  def seqOp(region: Region, c: Call, cm: Boolean, af: Double, afm: Boolean) {
    if (combiner != null && !cm && !afm) {
      combiner.merge(c, af)
    }
  }

  override def combOp(agg2: RegionValueAggregator) {
    val other = agg2.asInstanceOf[RegionValueInbreedingAggregator]
    if (other.combiner != null) {
      if (combiner == null)
        combiner = new InbreedingCombiner()
      combiner.merge(other.combiner)
    }
  }

  override def result(rvb: RegionValueBuilder) {
    if (combiner != null)
      combiner.result(rvb)
    else
      rvb.setMissing()
  }

  override def newInstance(): RegionValueInbreedingAggregator = new RegionValueInbreedingAggregator()

  override def copy(): RegionValueInbreedingAggregator = {
    val rva = new RegionValueInbreedingAggregator()
    rva.combiner = combiner.copy()
    rva
  }

  override def clear() {
    combiner.clear()
  }
}

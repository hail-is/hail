package is.hail.annotations.aggregators

import is.hail.annotations.{Region, RegionValueBuilder}
import is.hail.stats.InbreedingCombiner
import is.hail.variant.Call

object RegionValueInbreedingAggregator {
  def typ = InbreedingCombiner.signature
}

class RegionValueInbreedingAggregator extends RegionValueAggregator {
  var combiner: InbreedingCombiner = _

  def initOp(af: Double, missing: Boolean) {
    if (!missing) {
      combiner = new InbreedingCombiner(af)
    }
  }

  def seqOp(region: Region, c: Call, missing: Boolean) {
    if (combiner != null && !missing) {
      combiner.merge(c)
    }
  }

  override def combOp(agg2: RegionValueAggregator) {
    val other = agg2.asInstanceOf[RegionValueInbreedingAggregator]
    if (other.combiner != null) {
      if (combiner == null)
        combiner = new InbreedingCombiner(other.combiner.af)
      combiner.merge(other.combiner)
    }
  }

  override def result(rvb: RegionValueBuilder) {
    if (combiner != null)
      combiner.result(rvb)
    else
      rvb.setMissing()
  }

  override def copy(): RegionValueAggregator = new RegionValueInbreedingAggregator()

  override def clear() {
    combiner = null
  }
}

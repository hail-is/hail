package is.hail.annotations.aggregators

import is.hail.annotations._
import is.hail.utils._

class RegionValueCollectBooleanAggregator extends RegionValueAggregator {
  private val ab = new MissingBooleanArrayBuilder()

  def seqOp(region: Region, b: Boolean, missing: Boolean) {
    if (missing)
      ab.addMissing()
    else
      ab.add(b)
  }

  def combOp(agg2: RegionValueAggregator) {
    val other = agg2.asInstanceOf[RegionValueCollectBooleanAggregator]
    other.ab.foreach { _ =>
      ab.addMissing()
    } { (_, x) =>
      ab.add(x)
    }
  }

  def result(rvb: RegionValueBuilder) {
    ab.write(rvb)
  }

  def copy(): RegionValueCollectBooleanAggregator = new RegionValueCollectBooleanAggregator()

  def clear() {
    ab.clear()
  }
}

class RegionValueCollectIntAggregator extends RegionValueAggregator {
  private val ab = new MissingIntArrayBuilder()

  def seqOp(region: Region, x: Int, missing: Boolean) {
    if (missing)
      ab.addMissing()
    else
      ab.add(x)
  }

  def combOp(agg2: RegionValueAggregator) {
    val other = agg2.asInstanceOf[RegionValueCollectIntAggregator]
    other.ab.foreach { _ =>
      ab.addMissing()
    } { (_, x) =>
      ab.add(x)
    }
  }

  def result(rvb: RegionValueBuilder) {
    ab.write(rvb)
  }

  def copy(): RegionValueCollectIntAggregator = new RegionValueCollectIntAggregator()

  def clear() {
    ab.clear()
  }
}

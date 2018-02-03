package is.hail.annotations.aggregators

import is.hail.expr._
import is.hail.annotations._
import is.hail.utils._

import scala.collection.mutable.BitSet

class RegionValueCollectBooleanAggregator extends RegionValueAggregator {
  private val ab = new MissingBooleanArrayBuilder()

  def seqOp(b: Boolean, missing: Boolean) {
    if (missing)
      ab.addMissing()
    else if (b)
      ab.add(b)
  }

  def seqOp(region: Region, off: Long, missing: Boolean) {
    seqOp(region.loadBoolean(off), missing)
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
}

class RegionValueCollectIntAggregator extends RegionValueAggregator {
  private val ab = new MissingIntArrayBuilder()

  def seqOp(x: Int, missing: Boolean) {
    if (missing)
      ab.addMissing()
    else
      ab.add(x)
  }

  def seqOp(region: Region, off: Long, missing: Boolean) {
    seqOp(region.loadInt(off), missing)
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
}

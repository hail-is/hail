package is.hail.annotations.aggregators

import is.hail.annotations._
import is.hail.expr.RegionValueAggregator

class RegionValueMaxBooleanAggregator extends RegionValueAggregator {
  private var max: Boolean = false
  private var found: Boolean = false

  def seqOp(i: Boolean, missing: Boolean) {
    found = true
    if (i > max)
      max = i
  }

  def seqOp(region: Region, off: Long, missing: Boolean) {
    seqOp(region.loadBoolean(off), missing)
  }

  def combOp(agg2: RegionValueAggregator) {
    val that = agg2.asInstanceOf[RegionValueMaxBooleanAggregator]
    if (that.max > max)
      max = that.max
  }

  def result(region: Region): Long =
    // FIXME: missingness???
    if (found)
      region.appendBoolean(max)
    else
      ???

  def copy(): RegionValueMaxBooleanAggregator = new RegionValueMaxBooleanAggregator()
}

class RegionValueMaxIntAggregator extends RegionValueAggregator {
  private var max: Int = 0
  private var found: Boolean = false

  def seqOp(i: Int, missing: Boolean) {
    found = true
    if (i > max)
      max = i
  }

  def seqOp(region: Region, off: Long, missing: Boolean) {
    seqOp(region.loadInt(off), missing)
  }

  def combOp(agg2: RegionValueAggregator) {
    val that = agg2.asInstanceOf[RegionValueMaxIntAggregator]
    if (that.max > max)
      max = that.max
  }

  def result(region: Region): Long =
    // FIXME: missingness???
    if (found)
      region.appendInt(max)
    else
      ???

  def copy(): RegionValueMaxIntAggregator = new RegionValueMaxIntAggregator()
}

class RegionValueMaxLongAggregator extends RegionValueAggregator {
  private var max: Long = 0L
  private var found: Boolean = false

  def seqOp(i: Long, missing: Boolean) {
    found = true
    if (i > max)
      max = i
  }

  def seqOp(region: Region, off: Long, missing: Boolean) {
    seqOp(region.loadLong(off), missing)
  }

  def combOp(agg2: RegionValueAggregator) {
    val that = agg2.asInstanceOf[RegionValueMaxLongAggregator]
    if (that.max > max)
      max = that.max
  }

  def result(region: Region): Long =
    // FIXME: missingness???
    if (found)
      region.appendLong(max)
    else
      ???

  def copy(): RegionValueMaxLongAggregator = new RegionValueMaxLongAggregator()
}

class RegionValueMaxFloatAggregator extends RegionValueAggregator {
  private var max: Float = 0.0f
  private var found: Boolean = false

  def seqOp(i: Float, missing: Boolean) {
    found = true
    if (i > max)
      max = i
  }

  def seqOp(region: Region, off: Long, missing: Boolean) {
    seqOp(region.loadFloat(off), missing)
  }

  def combOp(agg2: RegionValueAggregator) {
    val that = agg2.asInstanceOf[RegionValueMaxFloatAggregator]
    if (that.max > max)
      max = that.max
  }

  def result(region: Region): Long =
    // FIXME: missingness???
    if (found)
      region.appendFloat(max)
    else
      ???

  def copy(): RegionValueMaxFloatAggregator = new RegionValueMaxFloatAggregator()
}

class RegionValueMaxDoubleAggregator extends RegionValueAggregator {
  private var max: Double = 0.0
  private var found: Boolean = false

  def seqOp(i: Double, missing: Boolean) {
    found = true
    if (i > max)
      max = i
  }

  def seqOp(region: Region, off: Long, missing: Boolean) {
    seqOp(region.loadDouble(off), missing)
  }

  def combOp(agg2: RegionValueAggregator) {
    val that = agg2.asInstanceOf[RegionValueMaxDoubleAggregator]
    if (that.max > max)
      max = that.max
  }

  def result(region: Region): Long =
    // FIXME: missingness???
    if (found)
      region.appendDouble(max)
    else
      ???

  def copy(): RegionValueMaxDoubleAggregator = new RegionValueMaxDoubleAggregator()
}

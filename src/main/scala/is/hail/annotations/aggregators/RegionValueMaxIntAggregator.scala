package is.hail.annotations.aggregators

import is.hail.annotations._

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

  def result(rvb: RegionValueBuilder) {
    if (found)
      rvb.addBoolean(max)
    else
      rvb.setMissing()
  }

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

  def result(rvb: RegionValueBuilder) {
    if (found)
      rvb.addInt(max)
    else
      rvb.setMissing()
  }

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

  def result(rvb: RegionValueBuilder) {
    if (found)
      rvb.addLong(max)
    else
      rvb.setMissing()
  }

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

  def result(rvb: RegionValueBuilder) {
    if (found)
      rvb.addFloat(max)
    else
      rvb.setMissing()
  }

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

  def result(rvb: RegionValueBuilder) {
    if (found)
      rvb.addDouble(max)
    else
      rvb.setMissing()
  }

  def copy(): RegionValueMaxDoubleAggregator = new RegionValueMaxDoubleAggregator()
}

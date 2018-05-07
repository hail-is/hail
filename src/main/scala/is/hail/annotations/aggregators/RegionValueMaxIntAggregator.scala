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

  def combOp(agg2: RegionValueAggregator) {
    val that = agg2.asInstanceOf[RegionValueMaxBooleanAggregator]
    if (that.max > max)
      max = that.max
    found = found | that.found
  }

  def result(rvb: RegionValueBuilder) {
    if (found)
      rvb.addBoolean(max)
    else
      rvb.setMissing()
  }

  def copy(): RegionValueMaxBooleanAggregator = new RegionValueMaxBooleanAggregator()

  def clear() {
    max = false
    found = false
  }
}

class RegionValueMaxIntAggregator extends RegionValueAggregator {
  private var max: Int = 0
  private var found: Boolean = false

  def seqOp(i: Int, missing: Boolean) {
    found = true
    if (i > max)
      max = i
  }

  def combOp(agg2: RegionValueAggregator) {
    val that = agg2.asInstanceOf[RegionValueMaxIntAggregator]
    if (that.max > max)
      max = that.max
    found = found | that.found
  }

  def result(rvb: RegionValueBuilder) {
    if (found)
      rvb.addInt(max)
    else
      rvb.setMissing()
  }

  def copy(): RegionValueMaxIntAggregator = new RegionValueMaxIntAggregator()

  def clear() {
    max = 0
    found = false
  }
}

class RegionValueMaxLongAggregator extends RegionValueAggregator {
  private var max: Long = 0L
  private var found: Boolean = false

  def seqOp(i: Long, missing: Boolean) {
    found = true
    if (i > max)
      max = i
  }

  def combOp(agg2: RegionValueAggregator) {
    val that = agg2.asInstanceOf[RegionValueMaxLongAggregator]
    if (that.max > max)
      max = that.max
    found = found | that.found
  }

  def result(rvb: RegionValueBuilder) {
    if (found)
      rvb.addLong(max)
    else
      rvb.setMissing()
  }

  def copy(): RegionValueMaxLongAggregator = new RegionValueMaxLongAggregator()

  def clear() {
    max = 0L
    found = false
  }
}

class RegionValueMaxFloatAggregator extends RegionValueAggregator {
  private var max: Float = 0.0f
  private var found: Boolean = false

  def seqOp(i: Float, missing: Boolean) {
    found = true
    if (i > max)
      max = i
  }

  def combOp(agg2: RegionValueAggregator) {
    val that = agg2.asInstanceOf[RegionValueMaxFloatAggregator]
    if (that.max > max)
      max = that.max
    found = found | that.found
  }

  def result(rvb: RegionValueBuilder) {
    if (found)
      rvb.addFloat(max)
    else
      rvb.setMissing()
  }

  def copy(): RegionValueMaxFloatAggregator = new RegionValueMaxFloatAggregator()

  def clear() {
    max = 0.0f
    found = false
  }
}

class RegionValueMaxDoubleAggregator extends RegionValueAggregator {
  private var max: Double = 0.0
  private var found: Boolean = false

  def seqOp(i: Double, missing: Boolean) {
    found = true
    if (i > max)
      max = i
  }

  def combOp(agg2: RegionValueAggregator) {
    val that = agg2.asInstanceOf[RegionValueMaxDoubleAggregator]
    if (that.max > max)
      max = that.max
    found = found | that.found
  }

  def result(rvb: RegionValueBuilder) {
    if (found)
      rvb.addDouble(max)
    else
      rvb.setMissing()
  }

  def copy(): RegionValueMaxDoubleAggregator = new RegionValueMaxDoubleAggregator()

  def clear() {
    max = 0.0
    found = false
  }
}

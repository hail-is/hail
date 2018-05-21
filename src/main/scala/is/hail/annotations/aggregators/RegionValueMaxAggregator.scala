package is.hail.annotations.aggregators

import is.hail.annotations._

class RegionValueMaxBooleanAggregator extends RegionValueAggregator {
  private var max: Boolean = false
  private var empty: Boolean = true

  def seqOp(region: Region, i: Boolean, missing: Boolean) {
    if (!missing && (i > max || empty)) {
      empty = false
      max = i
    }
  }

  def combOp(agg2: RegionValueAggregator) {
    val that = agg2.asInstanceOf[RegionValueMaxBooleanAggregator]
    if (!that.empty && (that.max > max || empty)) {
      empty = false
      max = that.max
    }
  }

  def result(rvb: RegionValueBuilder) {
    if (empty)
      rvb.setMissing()
    else
      rvb.addBoolean(max)
  }

  def copy(): RegionValueMaxBooleanAggregator = new RegionValueMaxBooleanAggregator()

  def clear() {
    max = false
    empty = true
  }
}

class RegionValueMaxIntAggregator extends RegionValueAggregator {
  private var max: Int = 0
  private var empty: Boolean = true

  def seqOp(region: Region, i: Int, missing: Boolean) {
    if (!missing && (i > max || empty)) {
      empty = false
      max = i
    }
  }

  def combOp(agg2: RegionValueAggregator) {
    val that = agg2.asInstanceOf[RegionValueMaxIntAggregator]
    if (!that.empty && (that.max > max || empty)) {
      empty = false
      max = that.max
    }
  }

  def result(rvb: RegionValueBuilder) {
    if (empty)
      rvb.setMissing()
    else
      rvb.addInt(max)
  }

  def copy(): RegionValueMaxIntAggregator = new RegionValueMaxIntAggregator()

  def clear() {
    max = 0
    empty = true
  }
}

class RegionValueMaxLongAggregator extends RegionValueAggregator {
  private var max: Long = 0L
  private var empty: Boolean = true

  def seqOp(region: Region, i: Long, missing: Boolean) {
    if (!missing && (i > max || empty)) {
      empty = false
      max = i
    }
  }

  def combOp(agg2: RegionValueAggregator) {
    val that = agg2.asInstanceOf[RegionValueMaxLongAggregator]
    if (!that.empty && (that.max > max || empty)) {
      empty = false
      max = that.max
    }
  }

  def result(rvb: RegionValueBuilder) {
    if (empty)
      rvb.setMissing()
    else
      rvb.addLong(max)
  }

  def copy(): RegionValueMaxLongAggregator = new RegionValueMaxLongAggregator()

  def clear() {
    max = 0L
    empty = true
  }
}

class RegionValueMaxFloatAggregator extends RegionValueAggregator {
  private var max: Float = 0.0f
  private var empty: Boolean = true

  def seqOp(region: Region, i: Float, missing: Boolean) {
    if (!missing && (i > max || empty)) {
      empty = false
      max = i
    }
  }

  def combOp(agg2: RegionValueAggregator) {
    val that = agg2.asInstanceOf[RegionValueMaxFloatAggregator]
    if (!that.empty && (that.max > max || empty)) {
      empty = false
      max = that.max
    }
  }

  def result(rvb: RegionValueBuilder) {
    if (empty)
      rvb.setMissing()
    else
      rvb.addFloat(max)
  }

  def copy(): RegionValueMaxFloatAggregator = new RegionValueMaxFloatAggregator()

  def clear() {
    max = 0.0f
    empty = true
  }
}

class RegionValueMaxDoubleAggregator extends RegionValueAggregator {
  private var max: Double = 0.0
  private var empty: Boolean = true

  def seqOp(region: Region, i: Double, missing: Boolean) {
    if (!missing && (i > max || empty)) {
      empty = false
      max = i
    }
  }

  def combOp(agg2: RegionValueAggregator) {
    val that = agg2.asInstanceOf[RegionValueMaxDoubleAggregator]
    if (!that.empty && (that.max > max || empty)) {
      empty = false
      max = that.max
    }
  }

  def result(rvb: RegionValueBuilder) {
    if (empty)
      rvb.setMissing()
    else
      rvb.addDouble(max)
  }

  def copy(): RegionValueMaxDoubleAggregator = new RegionValueMaxDoubleAggregator()

  def clear() {
    max = 0.0
    empty = true
  }
}

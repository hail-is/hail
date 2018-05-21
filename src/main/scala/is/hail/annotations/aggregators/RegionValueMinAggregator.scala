package is.hail.annotations.aggregators

import is.hail.annotations._

class RegionValueMinBooleanAggregator extends RegionValueAggregator {
  private var min: Boolean = false
  private var empty: Boolean = true

  def seqOp(region: Region, i: Boolean, missing: Boolean) {
    if (!missing && (i < min || empty)) {
      empty = false
      min = i
    }
  }

  def combOp(agg2: RegionValueAggregator) {
    val that = agg2.asInstanceOf[RegionValueMinBooleanAggregator]
    if (!that.empty && (that.min < min || empty)) {
      empty = false
      min = that.min
    }
  }

  def result(rvb: RegionValueBuilder) {
    if (empty)
      rvb.setMissing()
    else
      rvb.addBoolean(min)
  }

  def copy(): RegionValueMinBooleanAggregator = new RegionValueMinBooleanAggregator()

  def clear() {
    min = false
    empty = true
  }
}

class RegionValueMinIntAggregator extends RegionValueAggregator {
  private var min: Int = 0
  private var empty: Boolean = true

  def seqOp(region: Region, i: Int, missing: Boolean) {
    if (!missing && (i < min || empty)) {
      empty = false
      min = i
    }
  }

  def combOp(agg2: RegionValueAggregator) {
    val that = agg2.asInstanceOf[RegionValueMinIntAggregator]
    if (!that.empty && (that.min < min || empty)) {
      empty = false
      min = that.min
    }
  }

  def result(rvb: RegionValueBuilder) {
    if (empty)
      rvb.setMissing()
    else
      rvb.addInt(min)
  }

  def copy(): RegionValueMinIntAggregator = new RegionValueMinIntAggregator()

  def clear() {
    min = 0
    empty = true
  }
}

class RegionValueMinLongAggregator extends RegionValueAggregator {
  private var min: Long = 0L
  private var empty: Boolean = true

  def seqOp(region: Region, i: Long, missing: Boolean) {
    if (!missing && (i < min || empty)) {
      empty = false
      min = i
    }
  }

  def combOp(agg2: RegionValueAggregator) {
    val that = agg2.asInstanceOf[RegionValueMinLongAggregator]
    if (!that.empty && (that.min < min || empty)) {
      empty = false
      min = that.min
    }
  }

  def result(rvb: RegionValueBuilder) {
    if (empty)
      rvb.setMissing()
    else
      rvb.addLong(min)
  }

  def copy(): RegionValueMinLongAggregator = new RegionValueMinLongAggregator()

  def clear() {
    min = 0L
    empty = true
  }
}

class RegionValueMinFloatAggregator extends RegionValueAggregator {
  private var min: Float = 0.0f
  private var empty: Boolean = true

  def seqOp(region: Region, i: Float, missing: Boolean) {
    if (!missing && (i < min || empty)) {
      empty = false
      min = i
    }
  }

  def combOp(agg2: RegionValueAggregator) {
    val that = agg2.asInstanceOf[RegionValueMinFloatAggregator]
    if (!that.empty && (that.min < min || empty)) {
      empty = false
      min = that.min
    }
  }

  def result(rvb: RegionValueBuilder) {
    if (empty)
      rvb.setMissing()
    else
      rvb.addFloat(min)
  }

  def copy(): RegionValueMinFloatAggregator = new RegionValueMinFloatAggregator()

  def clear() {
    min = 0.0f
    empty = true
  }
}

class RegionValueMinDoubleAggregator extends RegionValueAggregator {
  private var min: Double = 0.0
  private var empty: Boolean = true

  def seqOp(region: Region, i: Double, missing: Boolean) {
    if (!missing && (i < min || empty)) {
      empty = false
      min = i
    }
  }

  def combOp(agg2: RegionValueAggregator) {
    val that = agg2.asInstanceOf[RegionValueMinDoubleAggregator]
    if (!that.empty && (that.min < min || empty)) {
      empty = false
      min = that.min
    }
  }

  def result(rvb: RegionValueBuilder) {
    if (empty)
      rvb.setMissing()
    else
      rvb.addDouble(min)
  }

  def copy(): RegionValueMinDoubleAggregator = new RegionValueMinDoubleAggregator()

  def clear() {
    min = 0.0
    empty = true
  }
}

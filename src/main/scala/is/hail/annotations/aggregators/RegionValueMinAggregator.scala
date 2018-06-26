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

  def newInstance(): RegionValueMinBooleanAggregator = new RegionValueMinBooleanAggregator()

  def copy(): RegionValueMinBooleanAggregator = {
    val rva = new RegionValueMinBooleanAggregator()
    rva.min = min
    rva.empty = empty
    rva
  }
  
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

  def newInstance(): RegionValueMinIntAggregator = new RegionValueMinIntAggregator()

  def copy(): RegionValueMinIntAggregator = {
    val rva = new RegionValueMinIntAggregator()
    rva.min = min
    rva.empty = empty
    rva
  }
  
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

  def newInstance(): RegionValueMinLongAggregator = new RegionValueMinLongAggregator()

  def copy(): RegionValueMinLongAggregator = {
    val rva = new RegionValueMinLongAggregator()
    rva.min = min
    rva.empty = empty
    rva
  }
  
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

  def newInstance(): RegionValueMinFloatAggregator = new RegionValueMinFloatAggregator()
  
  def copy(): RegionValueMinFloatAggregator = {
    val rva = new RegionValueMinFloatAggregator()
    rva.min = min
    rva.empty = empty
    rva
  }
  
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

  def newInstance(): RegionValueMinDoubleAggregator = new RegionValueMinDoubleAggregator()

  def copy(): RegionValueMinDoubleAggregator = {
    val rva = new RegionValueMinDoubleAggregator()
    rva.min = min
    rva.empty = empty
    rva
  }
  
  def clear() {
    min = 0.0
    empty = true
  }
}

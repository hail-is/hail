package is.hail.annotations.aggregators

import is.hail.annotations._
import is.hail.expr.types._
import is.hail.utils._

class RegionValueArraySumIntAggregator extends RegionValueAggregator {
  private var sum: Array[Int] = _
  private[this] val t = TArray(TInt32())

  def seqOp(region: Region, aoff: Long, missing: Boolean) {
    if (!missing)
      if (sum == null)
        sum = Array.tabulate(t.loadLength(region, aoff)) { i =>
          if (t.isElementDefined(region, aoff, i)) {
            region.loadInt(t.loadElement(region, aoff, i))
          } else 0
        }
      else {
        val len = t.loadLength(region, aoff)
        if (len != sum.length)
          fatal(
            s"""cannot aggregate arrays of unequal length with `sum'
               |Found conflicting arrays of size (${ sum.length }) and ($len)""".stripMargin)
        else {
          var i = 0
          while (i < len) {
            if (t.isElementDefined(region, aoff, i)) {
              sum(i) += region.loadInt(t.loadElement(region, aoff, i))
            }
            i += 1
          }
        }
      }
  }

  def combOp(_that: RegionValueAggregator) {
    val that = _that.asInstanceOf[RegionValueArraySumIntAggregator]
    if (that.sum != null && sum != null && that.sum.length != sum.length)
      fatal(
        s"""cannot aggregate arrays of unequal length with `sum'
               |Found conflicting arrays of size (${ sum.length })
               |and (${ that.sum.length })""".stripMargin)
    if (sum == null)
      sum = that.sum
    else if (that.sum != null) {
      var i = 0
      while (i < sum.length) {
        sum(i) += that.sum(i)
        i += 1
      }
    }
  }

  def result(rvb: RegionValueBuilder) {
    if (sum == null) {
      rvb.setMissing()
    } else {
      rvb.startArray(sum.length)
      sum.foreach(rvb.addInt)
      rvb.endArray()
    }
  }

  def copy(): RegionValueArraySumIntAggregator = new RegionValueArraySumIntAggregator()

  def clear() {
    sum = null
  }
}

class RegionValueArraySumLongAggregator extends RegionValueAggregator {
  private var sum: Array[Long] = _
  private[this] val t = TArray(TInt64())

  def seqOp(region: Region, aoff: Long, missing: Boolean) {
    if (!missing)
      if (sum == null)
        sum = Array.tabulate(t.loadLength(region, aoff)) { i =>
          if (t.isElementDefined(region, aoff, i)) {
            region.loadLong(t.loadElement(region, aoff, i))
          } else 0L
        }
      else {
        val len = t.loadLength(region, aoff)
        if (len != sum.length)
          fatal(
            s"""cannot aggregate arrays of unequal length with `sum'
               |Found conflicting arrays of size (${ sum.length }) and ($len)""".stripMargin)
        else {
          var i = 0
          while (i < len) {
            if (t.isElementDefined(region, aoff, i)) {
              sum(i) += region.loadLong(t.loadElement(region, aoff, i))
            }
            i += 1
          }
        }
      }
  }

  def combOp(_that: RegionValueAggregator) {
    val that = _that.asInstanceOf[RegionValueArraySumLongAggregator]
    if (that.sum != null && sum != null && that.sum.length != sum.length)
      fatal(
        s"""cannot aggregate arrays of unequal length with `sum'
               |Found conflicting arrays of size (${ sum.length })
               |and (${ that.sum.length })""".stripMargin)
    if (sum == null)
      sum = that.sum
    else if (that.sum != null) {
      var i = 0
      while (i < sum.length) {
        sum(i) += that.sum(i)
        i += 1
      }
    }
  }

  def result(rvb: RegionValueBuilder) {
    if (sum == null) {
      rvb.setMissing()
    } else {
      rvb.startArray(sum.length)
      sum.foreach(rvb.addLong)
      rvb.endArray()
    }
  }

  def copy(): RegionValueArraySumLongAggregator = new RegionValueArraySumLongAggregator()

  def clear() {
    sum = null
  }
}

class RegionValueArraySumFloatAggregator extends RegionValueAggregator {
  private var sum: Array[Float] = _
  private[this] val t = TArray(TFloat32())

  def seqOp(region: Region, aoff: Long, missing: Boolean) {
    if (!missing)
      if (sum == null)
        sum = Array.tabulate(t.loadLength(region, aoff)) { i =>
          if (t.isElementDefined(region, aoff, i)) {
            region.loadFloat(t.loadElement(region, aoff, i))
          } else 0
        }
      else {
        val len = t.loadLength(region, aoff)
        if (len != sum.length)
          fatal(
            s"""cannot aggregate arrays of unequal length with `sum'
               |Found conflicting arrays of size (${ sum.length }) and ($len)""".stripMargin)
        else {
          var i = 0
          while (i < len) {
            if (t.isElementDefined(region, aoff, i)) {
              sum(i) += region.loadFloat(t.loadElement(region, aoff, i))
            }
            i += 1
          }
        }
      }
  }

  def combOp(_that: RegionValueAggregator) {
    val that = _that.asInstanceOf[RegionValueArraySumFloatAggregator]
    if (that.sum != null && sum != null && that.sum.length != sum.length)
      fatal(
        s"""cannot aggregate arrays of unequal length with `sum'
               |Found conflicting arrays of size (${ sum.length })
               |and (${ that.sum.length })""".stripMargin)
    if (sum == null)
      sum = that.sum
    else if (that.sum != null) {
      var i = 0
      while (i < sum.length) {
        sum(i) += that.sum(i)
        i += 1
      }
    }
  }

  def result(rvb: RegionValueBuilder) {
    if (sum == null) {
      rvb.setMissing()
    } else {
      rvb.startArray(sum.length)
      sum.foreach(rvb.addFloat)
      rvb.endArray()
    }
  }

  def copy(): RegionValueArraySumFloatAggregator = new RegionValueArraySumFloatAggregator()

  def clear() {
    sum = null
  }
}

class RegionValueArraySumDoubleAggregator extends RegionValueAggregator {
  private var sum: Array[Double] = _
  private[this] val t = TArray(TFloat64())

  def seqOp(region: Region, aoff: Long, missing: Boolean) {
    if (!missing)
      if (sum == null)
        sum = Array.tabulate(t.loadLength(region, aoff)) { i =>
          if (t.isElementDefined(region, aoff, i)) {
            region.loadDouble(t.loadElement(region, aoff, i))
          } else 0
        }
      else {
        val len = t.loadLength(region, aoff)
        if (len != sum.length)
          fatal(
            s"""cannot aggregate arrays of unequal length with `sum'
               |Found conflicting arrays of size (${ sum.length }) and ($len)""".stripMargin)
        else {
          var i = 0
          while (i < len) {
            if (t.isElementDefined(region, aoff, i)) {
              sum(i) += region.loadDouble(t.loadElement(region, aoff, i))
            }
            i += 1
          }
        }
      }
  }

  def combOp(_that: RegionValueAggregator) {
    val that = _that.asInstanceOf[RegionValueArraySumDoubleAggregator]
    if (that.sum != null && sum != null && that.sum.length != sum.length)
      fatal(
        s"""cannot aggregate arrays of unequal length with `sum'
               |Found conflicting arrays of size (${ sum.length })
               |and (${ that.sum.length })""".stripMargin)
    if (sum == null)
      sum = that.sum
    else if (that.sum != null) {
      var i = 0
      while (i < sum.length) {
        sum(i) += that.sum(i)
        i += 1
      }
    }
  }

  def result(rvb: RegionValueBuilder) {
    if (sum == null) {
      rvb.setMissing()
    } else {
      rvb.startArray(sum.length)
      sum.foreach(rvb.addDouble)
      rvb.endArray()
    }
  }

  def copy(): RegionValueArraySumDoubleAggregator = new RegionValueArraySumDoubleAggregator()

  def clear() {
    sum = null
  }
}

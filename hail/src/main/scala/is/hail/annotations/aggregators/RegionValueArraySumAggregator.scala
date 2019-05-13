package is.hail.annotations.aggregators

import is.hail.annotations._
import is.hail.expr.types.virtual.{TArray, TFloat64, TInt64, Type}
import is.hail.utils._

class RegionValueArraySumLongAggregator(t: Type) extends RegionValueAggregator {
  private var sum: Array[Long] = _
  assert(t.isOfType(TArray(TInt64())))
  private val typ = t.asInstanceOf[TArray].physicalType

  def seqOp(region: Region, aoff: Long, missing: Boolean) {
    if (!missing)
      if (sum == null)
        sum = Array.tabulate(typ.loadLength(region, aoff)) { i =>
          if (typ.isElementDefined(region, aoff, i)) {
            region.loadLong(typ.loadElement(region, aoff, i))
          } else 0L
        }
      else {
        val len = typ.loadLength(region, aoff)
        if (len != sum.length)
          fatal(
            s"""cannot aggregate arrays of unequal length with 'sum'
               |Found conflicting arrays of size (${ sum.length }) and ($len)""".stripMargin)
        else {
          var i = 0
          while (i < len) {
            if (typ.isElementDefined(region, aoff, i)) {
              sum(i) += region.loadLong(typ.loadElement(region, aoff, i))
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
        s"""cannot aggregate arrays of unequal length with 'sum'
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

  def newInstance(): RegionValueArraySumLongAggregator = new RegionValueArraySumLongAggregator(t)

  def copy(): RegionValueArraySumLongAggregator = {
    val rva = new RegionValueArraySumLongAggregator(t)
    if (sum != null)
      rva.sum = sum.clone()
    rva
  }

  def clear() {
    sum = null
  }
}

class RegionValueArraySumDoubleAggregator(t: Type) extends RegionValueAggregator {
  private var sum: Array[Double] = _
  assert(t.isOfType(TArray(TFloat64())))
  private val typ = t.asInstanceOf[TArray].physicalType

  def seqOp(region: Region, aoff: Long, missing: Boolean) {
    if (!missing)
      if (sum == null)
        sum = Array.tabulate(typ.loadLength(region, aoff)) { i =>
          if (typ.isElementDefined(region, aoff, i)) {
            region.loadDouble(typ.loadElement(region, aoff, i))
          } else 0
        }
      else {
        val len = typ.loadLength(region, aoff)
        if (len != sum.length)
          fatal(
            s"""cannot aggregate arrays of unequal length with 'sum'
               |Found conflicting arrays of size (${ sum.length }) and ($len)""".stripMargin)
        else {
          var i = 0
          while (i < len) {
            if (typ.isElementDefined(region, aoff, i)) {
              sum(i) += region.loadDouble(typ.loadElement(region, aoff, i))
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
        s"""cannot aggregate arrays of unequal length with 'sum'
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

  def newInstance(): RegionValueArraySumDoubleAggregator = new RegionValueArraySumDoubleAggregator(t)

  def copy(): RegionValueArraySumDoubleAggregator = {
    val rva = new RegionValueArraySumDoubleAggregator(t)
    if (sum != null)
      rva.sum = sum.clone()
    rva
  }

  def clear() {
    sum = null
  }
}

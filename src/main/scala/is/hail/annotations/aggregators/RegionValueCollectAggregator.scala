package is.hail.annotations.aggregators

import is.hail.expr._
import is.hail.annotations._
import is.hail.utils._

import scala.collection.mutable.BitSet

class RegionValueCollectBooleanAggregator extends RegionValueAggregator {
  private var length = 0
  private val elements = new BitSet()
  private val isMissing = new BitSet()

  def seqOp(b: Boolean, missing: Boolean) {
    if (missing)
      isMissing.add(length)
    else if (b)
      elements.add(length)
    length += 1
  }

  def seqOp(region: Region, off: Long, missing: Boolean) {
    seqOp(region.loadBoolean(off), missing)
  }

  def combOp(agg2: RegionValueAggregator) {
    val other = agg2.asInstanceOf[RegionValueCollectBooleanAggregator]
    var i = 0
    while (i < other.length) {
      if (other.isMissing(i))
        isMissing.add(i)
      else if (other.elements(i))
        elements.add(length + i)
      i += 1
    }
    length += other.length
  }

  private val rvb = new RegionValueBuilder()

  private val typ = TArray(TBoolean())

  def result(region: Region): Long = {
    rvb.set(region)
    rvb.start(typ)
    rvb.startArray(length)
    var i = 0
    while (i < length) {
      if (isMissing(i))
        rvb.setMissing()
      else
        rvb.addBoolean(elements(i))
      i += 1
    }
    rvb.endArray()
    rvb.end()
  }

  def copy(): RegionValueCollectBooleanAggregator = new RegionValueCollectBooleanAggregator()
}

class RegionValueCollectIntAggregator extends RegionValueAggregator {
  private var length = 0
  private val elements = new ArrayBuilder[Int]()
  private val isMissing = new BitSet()

  def seqOp(b: Int, missing: Boolean) {
    if (missing)
      isMissing.add(length)
    else
      elements += (b)
    length += 1
  }

  def seqOp(region: Region, off: Long, missing: Boolean) {
    seqOp(region.loadInt(off), missing)
  }

  def combOp(agg2: RegionValueAggregator) {
    val other = agg2.asInstanceOf[RegionValueCollectIntAggregator]
    var i = 0
    var j = 0
    while (i < other.length) {
      if (isMissing(i))
        isMissing.add(length + i)
      else {
        elements += other.elements(j)
        j += 1
      }
      i += 1
    }
    length += other.length
  }

  private val rvb = new RegionValueBuilder()

  private val typ = TArray(TInt32())

  def result(region: Region): Long = {
    rvb.set(region)
    rvb.start(typ)
    rvb.startArray(length)
    var i = 0
    var j = 0
    while (i < length) {
      if (isMissing(i))
        rvb.setMissing()
      else {
        rvb.addInt(elements(j))
        j += 1
      }
      i += 1
    }
    rvb.endArray()
    rvb.end()
  }

  def copy(): RegionValueCollectIntAggregator = new RegionValueCollectIntAggregator()
}

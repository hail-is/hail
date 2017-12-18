package is.hail.annotations.aggregators

import is.hail.annotations._
import is.hail.expr.RegionValueAggregator

class RegionValueTakeBooleanAggregator(n: Int) extends RegionValueAggregator {
  private var a: Array[Boolean] = new Array(n)
  private var filled: Int = 0

  def seqOp(x: Boolean, missing: Boolean) {
    if (filled < n) {
      a(filled) = x
      filled += 1
    }
  }

  def seqOp(region: Region, off: Long, missing: Boolean) {
    seqOp(region.loadBoolean(off), missing)
  }

  def combOp(agg2: RegionValueAggregator) {
    val that = agg2.asInstanceOf[RegionValueTakeBooleanAggregator]
    var i = 0
    while (filled < n) {
      a(filled) = that.a(i)
      filled += 1
      i += 1
    }
  }

  val typ = TArray(TBoolean())
  private val rvb = new RegionValueBuilder()

  def result(region: Region): Long = {
    rvb.start(newRoot: Type)
  }

  def copy(): RegionValueTakeBooleanAggregator = new RegionValueTakeBooleanAggregator(n)
}

class RegionValueTakeIntAggregator extends RegionValueAggregator {

}

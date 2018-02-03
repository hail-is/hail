package is.hail.annotations.aggregators

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.types._
import org.apache.spark.util.StatCounter

object RegionValueStatisticsAggregator {
  val typ: TStruct = TStruct(
    "mean" -> TFloat64(),
    "stddev" -> TFloat64(),
    "min" -> TFloat64(), // FIXME should really preserve the input type
    "max" -> TFloat64(), // FIXME should really preserve the input type
    "count" -> TInt64(),
    "sum" -> TFloat64())
}

class RegionValueStatisticsAggregator extends RegionValueAggregator {
  private val sc = new StatCounter()

  def seqOp(x: Double, missing: Boolean) {
    if (!missing)
      sc.merge(x)
  }

  def seqOp(region: Region, off: Long, missing: Boolean) {
    seqOp(region.loadDouble(off), missing)
  }

  def combOp(agg2: RegionValueAggregator) {
    sc.merge(agg2.asInstanceOf[RegionValueStatisticsAggregator].sc)
  }

  def result(rvb: RegionValueBuilder) {
    if (sc.count == 0)
      rvb.setMissing()
    else {
      rvb.startStruct()
      rvb.addDouble(sc.mean)
      rvb.addDouble(sc.stdev)
      rvb.addDouble(sc.min)
      rvb.addDouble(sc.max)
      rvb.addLong(sc.count)
      rvb.addDouble(sc.sum)
      rvb.endStruct()
    }
  }

  def copy() = new RegionValueStatisticsAggregator()
}

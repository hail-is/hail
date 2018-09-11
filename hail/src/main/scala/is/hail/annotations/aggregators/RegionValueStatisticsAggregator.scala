package is.hail.annotations.aggregators

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.types._
import org.apache.spark.util.StatCounter

object RegionValueStatisticsAggregator {
  val typ: TStruct = TStruct(
    "mean" -> TFloat64(),
    "stdev" -> TFloat64(),
    "min" -> TFloat64(), // FIXME should really preserve the input type
    "max" -> TFloat64(), // FIXME should really preserve the input type
    "n" -> TInt64(),
    "sum" -> TFloat64())
}

class RegionValueStatisticsAggregator extends RegionValueAggregator {
  private var sc = new StatCounter()

  def seqOp(region: Region, x: Double, missing: Boolean) {
    if (!missing)
      sc.merge(x)
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

  def newInstance() = new RegionValueStatisticsAggregator()

  def copy(): RegionValueStatisticsAggregator = {
    val rva = new RegionValueStatisticsAggregator()
    rva.sc = sc.copy()
    rva
  }

  def clear() {
    sc = new StatCounter()
  }
}

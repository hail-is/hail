package is.hail.annotations.aggregators

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr._
import org.apache.spark.util.StatCounter

object RegionValueStatisticsAggregator {
  val typ: TStruct = TStruct(
    "mean" -> TFloat64(),
    "stddev" -> TFloat64(),
    "min" -> TFloat64(), // FIXME should really preserve the input type
    "max" -> TFloat64(),
    "count" -> TInt64(),
    "sum" -> TFloat64())

  private val fb = FunctionBuilder.functionBuilder[Region, StatCounter, Long]
  private val statCounter = fb.getArg[StatCounter](2)
  private val srvb = new StagedRegionValueBuilder(fb, typ)
  fb.emit(Code(srvb.start(),
    srvb.addDouble(statCounter.invoke("mean")),
    srvb.addDouble(statCounter.invoke("stddev")),
    srvb.addDouble(statCounter.invoke("min")),
    srvb.addDouble(statCounter.invoke("max")),
    srvb.addLong(statCounter.invoke("count")),
    srvb.addDouble(statCounter.invoke("sum")),
    srvb.returnStart()))

  private val makeF = fb.result()
  def result(region: Region, sc: StatCounter): Long =
    makeF()(region, sc)
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

  def result(region: Region): Long =
    // FIXME: divide by zero should be NA not NaN
    RegionValueStatisticsAggregator.result(region, sc)

  def copy() = new RegionValueStatisticsAggregator()
}

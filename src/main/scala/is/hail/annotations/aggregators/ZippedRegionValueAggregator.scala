package is.hail.annotations.aggregators

import is.hail.expr._
import is.hail.annotations._

/**
  * This class gaurantees that the fields of the resulting struct are numbered
  * in the same order as the array of aggregators given to the constructor
  *
  **/
class ZippedRegionValueAggregator(val aggs: Array[RegionValueAggregator]) extends RegionValueAggregator {

  private val fields = aggs.map(_.typ).zipWithIndex.map { case (t, i) => (i.toString -> t) }

  val typ: TStruct = TStruct(true, fields:_*)

  def seqOp(region: MemoryBuffer, off: Long, missing: Boolean) {
    aggs.foreach(_.seqOp(region, off, missing))
  }

  def combOp(agg2: RegionValueAggregator) {
    (aggs zip agg2.asInstanceOf[ZippedRegionValueAggregator].aggs)
      .foreach { case (l,r) => l.combOp(r) }
  }

  def result(region: MemoryBuffer): Long = {
    val rvb = new RegionValueBuilder(region)
    rvb.start(typ)
    rvb.startStruct()
    aggs.foreach(agg => rvb.addRegionValue(agg.typ, region, agg.result(region)))
    rvb.endStruct()
    rvb.end()
  }

  def copy(): ZippedRegionValueAggregator =
    new ZippedRegionValueAggregator(aggs.map(_.copy))
}


package is.hail.annotations.aggregators

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr._
import is.hail.expr.ir._
import is.hail.utils._

class TransformedRegionValueAggregator(
  seqOp: () => AsmFunction4[MemoryBuffer, Long, Boolean, RegionValueAggregator, Unit],
  val next: RegionValueAggregator) extends RegionValueAggregator {

  def seqOp(region: MemoryBuffer, offset: Long, missing: Boolean) {
    seqOp()(region, offset, missing, next)
  }

  def combOp(agg2: RegionValueAggregator) {
    next.combOp(agg2.asInstanceOf[TransformedRegionValueAggregator].next)
  }

  def result(region: MemoryBuffer): Long = {
    next.result(region)
  }

  def copy(): TransformedRegionValueAggregator =
    new TransformedRegionValueAggregator(seqOp, next)
}


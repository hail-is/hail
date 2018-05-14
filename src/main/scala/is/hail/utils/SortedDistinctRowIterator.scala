package is.hail.utils

import is.hail.annotations._
import is.hail.rvd.{OrderedRVDType, RVDContext}

object SortedDistinctRowIterator {
  def transformer(ort: OrderedRVDType): (RVDContext, Iterator[RegionValue]) => SortedDistinctRowIterator =
    (ctx, it) => new SortedDistinctRowIterator(ort, it, ctx)
}

class SortedDistinctRowIterator(ort: OrderedRVDType, it: Iterator[RegionValue], ctx: RVDContext) extends Iterator[RegionValue] {
  private val bit = it.buffered
  private val wrv: WritableRegionValue = WritableRegionValue(ort.rowType, ctx.freshRegion)

  override def hasNext: Boolean = bit.hasNext

  override def next(): RegionValue = {
    wrv.set(bit.next())
    while (bit.hasNext && ort.kInRowOrd.compare(wrv.value, bit.head) == 0)
      bit.next()
    wrv.value
  }
}

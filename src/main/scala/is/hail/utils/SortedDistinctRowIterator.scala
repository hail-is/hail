package is.hail.utils

import is.hail.annotations._
import is.hail.rvd.OrderedRVDType

object SortedDistinctRowIterator {
  def transformer(ort: OrderedRVDType): Iterator[RegionValue] => SortedDistinctRowIterator =
    new SortedDistinctRowIterator(ort, _)
}

class SortedDistinctRowIterator(ort: OrderedRVDType, it: Iterator[RegionValue]) extends Iterator[RegionValue] {
  private val bit = it.buffered
  private val wrv: WritableRegionValue = WritableRegionValue(ort.rowType)

  override def hasNext: Boolean = bit.hasNext

  override def next(): RegionValue = {
    wrv.set(bit.next())
    while (bit.hasNext && ort.kInRowOrd.compare(wrv.value, bit.head) == 0)
      bit.next()
    wrv.value
  }
}

package is.hail.utils

import is.hail.expr._
import is.hail.annotations._
import is.hail.sparkextras._

object SortedDistinctRowIterator {
  def transformer(ort: OrderedRDD2Type): Iterator[RegionValue] => SortedDistinctRowIterator =
    new SortedDistinctRowIterator(ort, _)
}

class SortedDistinctRowIterator(ort: OrderedRDD2Type, it: Iterator[RegionValue]) extends Iterator[RegionValue] {
  private val bit = it.buffered
  private val wrv: WritableRegionValue = WritableRegionValue(ort.rowType)
  private var haveNext = false
  private var stale = false

  if (bit.hasNext) {
    haveNext = true
    wrv.set(bit.next())
  }

  private def advance() {
    while (bit.hasNext && ort.kInRowOrd.compare(wrv.value, bit.head) == 0)
      bit.next()
    haveNext = bit.hasNext
    if (haveNext)
      wrv.set(bit.next())
    stale = false
  }

  override def hasNext: Boolean = {
    if (stale)
      advance()
    haveNext
  }

  override def next(): RegionValue = {
    if (stale)
      advance()
    stale = true
    assert(haveNext)
    wrv.value
  }
}

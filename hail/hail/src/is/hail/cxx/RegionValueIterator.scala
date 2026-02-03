package is.hail.cxx

import is.hail.annotations.RegionValue

class RegionValueIterator(it: Iterator[RegionValue]) extends Iterator[Long] {

  override def next(): Long = it.next().offset

  override def hasNext: Boolean = it.hasNext
}

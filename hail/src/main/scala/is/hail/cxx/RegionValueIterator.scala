package is.hail.cxx

import is.hail.annotations.RegionValue

class RegionValueIterator(it: Iterator[RegionValue]) extends Iterator[Long] {

  def next(): Long = it.next().offset

  def hasNext: Boolean = it.hasNext
}

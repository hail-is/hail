package is.hail.nativecode

import is.hail.annotations.RegionValue

class RegionValueIterator(it: Iterator[RegionValue]) {

  def hasNext: Boolean = it.hasNext

  def next(): Long = it.next().offset

}

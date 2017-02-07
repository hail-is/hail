package is.hail.utils

class SortedUnionPairIterator[K, V](it1: Iterator[(K, V)], it2: Iterator[(K, V)])
  (implicit ord: Ordering[K]) extends Iterator[(K, V)] {
  private val bit1 = it1.buffered
  private val bit2 = it2.buffered
  import ord._

  def hasNext(): Boolean = bit1.hasNext || bit2.hasNext

  def next(): (K, V) = {
    if (bit1.isEmpty)
      bit2.next()
    else if (bit2.isEmpty)
      bit1.next()
    else if (bit1.head._1 < bit2.head._1)
      bit1.next()
    else
      bit2.next()
  }
}
package is.hail.utils

class SortedDistinctPairIterator[K, V](it: Iterator[(K, V)]) extends Iterator[(K, V)] {
  val bit = it.buffered

  override def hasNext: Boolean = bit.hasNext

  override def next(): (K, V) = {
    val (k, v) = bit.next()
    while (bit.hasNext && bit.head._1 == k) {
      bit.next()
    }
    (k, v)
  }
}

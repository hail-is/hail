package is.hail.utils

class SortedDistinctPairIterator[K, V](it: Iterator[(K, V)], f: K => Unit = (k: K) => ()) extends Iterator[(K, V)] {
  val bit = it.buffered
  var i = 0

  override def hasNext: Boolean = bit.hasNext

  override def next(): (K, V) = {
    val (k, v) = bit.next()
    while (bit.hasNext && bit.head._1 == k) {
      f(k)
      bit.next()
    }
    (k, v)
  }
}

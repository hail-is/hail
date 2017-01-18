package is.hail.utils

import scala.collection.mutable.ListBuffer

class SpanningIterator[K, V](val it: Iterator[(K, V)]) extends Iterator[(K, Iterable[V])] {
  val bit = it.buffered
  var n: Option[(K, Iterable[V])] = None

  override def hasNext: Boolean = {
    if (n.isDefined) return true
    n = computeNext
    n.isDefined
  }

  override def next(): ((K, Iterable[V])) = {
    val result = n.get
    n = None
    result
  }

  def computeNext: (Option[(K, Iterable[V])]) = {
    var k: Option[K] = None
    val span: ListBuffer[V] = ListBuffer()
    while (bit.hasNext) {
      if (k.isEmpty) {
        val (k_, v_) = bit.next
        k = Some(k_)
        span += v_
      } else if (bit.head._1 == k.get) {
        span += bit.next._2
      } else {
        return Some((k.get, span))
      }
    }
    k.map((_, span))
  }
}

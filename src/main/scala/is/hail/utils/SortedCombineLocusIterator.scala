package is.hail.utils

import is.hail.variant.Locus

import scala.collection.mutable.ArrayBuffer


class SortedCombineLocusIterator[T](it: Iterator[(Locus, T)]) extends Iterator[(Locus, IndexedSeq[T])] {

  private val bit = it.buffered

  private val resBuilder = new ArrayBuffer[T]()

  def hasNext: Boolean = {
    bit.nonEmpty
  }

  def next(): (Locus, IndexedSeq[T]) = {
    val (l, t) = bit.next()
    resBuilder.clear()
    resBuilder += t
    while (bit.hasNext && bit.head._1.compare(l) == 0) {
      resBuilder += bit.next()._2
    }
    (l, resBuilder.result(): IndexedSeq[T])
  }
}

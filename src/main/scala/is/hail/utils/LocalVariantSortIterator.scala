package is.hail.utils

import is.hail.variant.Variant

import scala.collection.mutable

object LocalVariantSortIterator {
  def apply[T](it: Iterator[(Variant, T)], maxShift: Int): Iterator[(Variant, T)] = {
    require(maxShift > 0, "max shift must be positive")

    val ord: Ordering[(Variant, T)] = new Ordering[(Variant, T)] {
      def compare(x: (Variant, T), y: (Variant, T)): Int = x._1.compare(y._1)
    }

    new AssertSortedIterator(
      new LocalVariantSortIterator(it, maxShift)(ord),
      maxShift)(ord)
  }
}

private class AssertSortedIterator[T](it: Iterator[T], maxShift: Int)(implicit ord: Ordering[T]) extends Iterator[T] {
  private val bit = it.buffered


  def hasNext(): Boolean = bit.hasNext

  def next(): T = {
    val t = bit.next()
    if (bit.hasNext && ord.compare(bit.head, t) < 0)
      fatal(s"provided maximum position shift `$maxShift' did not produce a sorted variant stream. Set higher max shift.")
    t
  }
}

class LocalVariantSortIterator[T] private(it: Iterator[(Variant, T)], maxShift: Int)
  (implicit ord: Ordering[(Variant, T)]) extends Iterator[(Variant, T)] {
  /**
    * This class sorts KV pairs of variants.  It assumes that the iterator
    * is close to sorted, and takes as a parameter the maximum number of
    * bases by which a variant can be out of order.  Contigs are assumed
    * to be completely sorted.
    */

  private val bit = it.buffered

  private val pq = mutable.PriorityQueue.empty[(Variant, T)](ord.reverse)

  def hasNext: Boolean = {
    pq.nonEmpty || bit.nonEmpty
  }

  def next(): (Variant, T) = {
    if (bit.isEmpty)
      pq.dequeue()
    else if (pq.isEmpty) {
      pq.enqueue(bit.next())
      next()
    } else {
      val vh = bit.head._1
      val ph = pq.head._1

      if (vh.contig != ph.contig)
        pq.dequeue()
      else if (vh.start - ph.start > maxShift)
        pq.dequeue()
      else {
        pq.enqueue(bit.next())
        next()
      }
    }
  }
}
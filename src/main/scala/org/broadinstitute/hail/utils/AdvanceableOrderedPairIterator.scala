package org.broadinstitute.hail.utils

trait AdvanceableOrderedPairIterator[K, V] extends Iterator[(K, V)] {self =>
  implicit val kOrdering: Ordering[K]

  def advanceTo(k: K): Unit

  override def buffered: BufferedAdvanceableOrderedPairIterator[K, V] =
    new BufferedAdvanceableOrderedPairIterator[K, V] {
      var hd: (K, V) = _
      var hdDefined: Boolean = false

      def head: (K, V) = {
        if (!hdDefined) {
          hd = self.next()
          hdDefined = true
        }
        hd
      }

      override def buffered = this

      implicit val kOrdering = self.kOrdering

      def hasNext = hdDefined || self.hasNext

      def next(): (K, V) = {
        if (hdDefined) {
          hdDefined = false
          hd
        } else
          self.next()
      }

      def advanceTo(k: K) {
        import Ordering.Implicits._

        if (!hdDefined
          || hd._1 < k) {
          hdDefined = false
          self.advanceTo(k)
        }
      }
    }
}

trait BufferedAdvanceableOrderedPairIterator[K, V] extends AdvanceableOrderedPairIterator[K, V]
  with BufferedIterator[(K, V)]

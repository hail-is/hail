package is.hail.utils.richUtils

import is.hail.utils.AdvanceableOrderedPairIterator

import scala.collection.mutable
import scala.reflect.ClassTag

class RichPairIterator[K, V](val it: Iterator[(K, V)]) {

  def sortedLeftJoinDistinct[V2](right: Iterator[(K, V2)])(implicit kOrd: Ordering[K]): Iterator[(K, (V, Option[V2]))] = {
    import Ordering.Implicits._

    val rightAdvanceable = new AdvanceableOrderedPairIterator[K, V2] {
      val bright = right.buffered

      val kOrdering = kOrd

      def hasNext = bright.hasNext

      def next() = bright.next()

      def advanceTo(k: K) {
        while (bright.hasNext && bright.head._1 < k)
          bright.next()
      }
    }

    sortedLeftJoinDistinct(rightAdvanceable)
  }

  def sortedLeftJoinDistinct[V2](right: AdvanceableOrderedPairIterator[K, V2]): Iterator[(K, (V, Option[V2]))] = {
    val bright = right.buffered

    new Iterator[(K, (V, Option[V2]))] {
      def hasNext = it.hasNext

      def next() = {
        val (k, v) = it.next()

        bright.advanceTo(k)
        if (bright.hasNext && bright.head._1 == k) {

          (k, (v, Some(bright.head._2)))
        } else
          (k, (v, None))
      }
    }
  }
}
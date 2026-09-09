package is.hail.collection.implicits

import scala.math.Ordering.Implicits._

class RichOrderedSeq[T](val s: Seq[T]) extends AnyVal {

  def isIncreasing(implicit ord: Ordering[T]): Boolean =
    s.isEmpty || s.lazyZip(s.tail).forall(_ < _)

  def isSorted(implicit ord: Ordering[T]): Boolean = s.isEmpty || s.lazyZip(s.tail).forall(_ <= _)
}

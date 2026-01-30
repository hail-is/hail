package is.hail.collection.implicits

import scala.collection.compat._

class RichOrderedSeq[T](val s: Seq[T]) extends AnyVal {

  import scala.math.Ordering.Implicits._

  def isIncreasing(implicit ord: Ordering[T]): Boolean =
    s.isEmpty || s.lazyZip(s.tail).forall(_ < _)

  def isSorted(implicit ord: Ordering[T]): Boolean = s.isEmpty || s.lazyZip(s.tail).forall(_ <= _)
}

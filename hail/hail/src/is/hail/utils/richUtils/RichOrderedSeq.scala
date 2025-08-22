package is.hail.utils.richUtils

import scala.collection.compat._

class RichOrderedSeq[T: Ordering](s: Seq[T]) {

  import scala.math.Ordering.Implicits._

  def isIncreasing: Boolean = s.isEmpty || s.lazyZip(s.tail).forall(_ < _)

  def isSorted: Boolean = s.isEmpty || s.lazyZip(s.tail).forall(_ <= _)
}

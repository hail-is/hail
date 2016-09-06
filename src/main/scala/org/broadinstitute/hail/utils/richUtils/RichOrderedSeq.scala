package org.broadinstitute.hail.utils.richUtils

class RichOrderedSeq[T: Ordering](s: Seq[T]) {

  import scala.math.Ordering.Implicits._

  def isIncreasing: Boolean = s.isEmpty || (s, s.tail).zipped.forall(_ < _)

  def isSorted: Boolean = s.isEmpty || (s, s.tail).zipped.forall(_ <= _)
}

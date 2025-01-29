package is.hail.utils.richUtils

import is.hail.utils._

class RichOrderedArray[T: Ordering](a: Array[T]) {
  def isIncreasing: Boolean = a.toSeq.isIncreasing

  def isSorted: Boolean = a.toSeq.isSorted
}

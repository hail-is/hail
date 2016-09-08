package org.broadinstitute.hail.utils.richUtils

import org.broadinstitute.hail.utils._

class RichOrderedArray[T: Ordering](a: Array[T]) {
  def isIncreasing: Boolean = a.toSeq.isIncreasing

  def isSorted: Boolean = a.toSeq.isSorted
}

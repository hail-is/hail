package is.hail.collection.implicits

class RichOrderedArray[T: Ordering](a: Array[T]) {
  def isIncreasing: Boolean = a.toSeq.isIncreasing

  def isSorted: Boolean = a.toSeq.isSorted
}

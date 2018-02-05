package is.hail.utils

class OrderedZipJoinIterator[T, U](
    left: BufferedIterator[T],
    leftDefault: T,
    right: BufferedIterator[U],
    rightDefault: U,
    odering: (T, U) => Int)
  extends Iterator[Muple[T, U]] {

  val muple = Muple(leftDefault, rightDefault)

  def hasNext: Boolean = left.hasNext || right.hasNext

  def next(): Muple[T, U] = {
    val c = {
      if (left.hasNext) {
        if (right.hasNext)
          odering(left.head, right.head)
        else
          -1
      } else if (right.hasNext)
          1
        else {
          assert(!hasNext)
          throw new NoSuchElementException("next on empty iterator")
        }
    }
    if (c == 0) {
      muple._1 = left.next()
      muple._2 = right.next()
    } else if (c < 0) {
      muple._1 = left.next()
      muple._2 = rightDefault
    } else {
      // c > 0
      muple._1 = leftDefault
      muple._2 = right.next()
    }
    muple
  }
}

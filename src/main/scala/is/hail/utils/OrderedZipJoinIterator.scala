package is.hail.utils

class OrderedZipJoinIterator[T, U](
    left: BufferedIterator[T],
    leftDefault: T,
    right: BufferedIterator[U],
    rightDefault: U,
    ordering: (T, U) => Int)
  extends Iterator[Muple[T, U]] {

  val muple = Muple(leftDefault, rightDefault)

  def hasNext: Boolean = left.hasNext || right.hasNext

  def next(): Muple[T, U] = {
    val c = {
      if (left.hasNext) {
        if (right.hasNext)
          ordering(left.head, right.head)
        else
          -1
      } else if (right.hasNext)
          1
        else {
          assert(!hasNext)
          throw new NoSuchElementException("next on empty iterator")
        }
    }
    if (c == 0)
      muple.set(left.next(), right.next())
    else if (c < 0)
      muple.set(left.next(), rightDefault)
    else
      // c > 0
      muple.set(leftDefault, right.next())
    muple
  }
}

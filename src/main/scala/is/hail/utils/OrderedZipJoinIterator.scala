package is.hail.utils

class OrderedZipJoinIterator[T, U](
    left: StagingIterator[T],
    leftDefault: T,
    right: StagingIterator[U],
    rightDefault: U,
    ordering: (T, U) => Int)
  extends StagingIterator[Muple[T, U]] {
  val head = Muple(leftDefault, rightDefault)
  var isValid: Boolean = true
  def advanceHead() {
    left.stage()
    right.stage()
    val c = {
      if (left.isValid) {
        if (right.isValid)
          ordering(left.value, right.value)
        else
          -1
      } else if (right.isValid)
          1
        else {
          isValid = false
          return
        }
    }
    if (c == 0)
      head.set(left.consume(), right.consume())
    else if (c < 0)
      head.set(left.consume(), rightDefault)
    else
      // c > 0
      head.set(leftDefault, right.consume())
  }
  advanceHead()
}

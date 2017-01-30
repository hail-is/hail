package is.hail.utils

abstract class IntIterator extends Iterator[Int] {
  def nextInt(): Int
  override def next(): Int = nextInt()

  def countNonNegative(): Int = {
    var cnt = 0
    while (hasNext) {
      if (nextInt() >= 0) cnt += 1
    }
    cnt
  }
}

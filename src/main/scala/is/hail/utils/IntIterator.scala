package is.hail.utils

abstract class IntIterator extends Iterator[Int] {
  def nextInt(): Int
  override def next(): Int = nextInt()

  override def count(p: Int => Boolean): Int = {
    var cnt = 0
    while (hasNext) {
      if (p(nextInt())) cnt += 1
    }
    cnt
  }
}

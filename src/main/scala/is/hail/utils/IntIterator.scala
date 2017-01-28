package is.hail.utils

abstract class IntIterator extends Iterator[Int] {
  def nextInt(): Int
  override def next(): Int = nextInt()
}

class GenericIntIterator(val it: Iterator[Int]) extends IntIterator {
  override def hasNext: Boolean = it.hasNext

  override def nextInt(): Int = it.next()
}
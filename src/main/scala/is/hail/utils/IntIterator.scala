package is.hail.utils

abstract class IntIterator extends Iterator[Int] {
  def nextInt(): Int
  override def next(): Int = nextInt()
}

package is.hail.utils

abstract class IteratorInt extends Iterator[Int] {
  def nextInt(): Int
  override def next(): Int = nextInt()
}

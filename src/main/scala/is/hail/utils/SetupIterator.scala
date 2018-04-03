package is.hail.utils

class SetupIterator[T](it: Iterator[T], setup: () => Unit) extends Iterator[T] {
  def hasNext: Boolean = it.hasNext
  def next(): T = {
    setup()
    it.next()
  }
}
